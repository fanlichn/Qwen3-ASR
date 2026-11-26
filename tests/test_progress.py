import logging
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.progress import TaskProgress, current_task
from app.qwen_asr import QwenTranscriber

KEY = 'test-only-not-a-production-key-1234567890'


class FakeEngine:
    def load(self):
        pass

    def transcribe(self, path, **kwargs):
        assert current_task.get() is not None
        return {'text': 'PRIVATE_TRANSCRIPT', 'language': 'zh', 'duration': 2,
                'segments': [{'start': 0, 'end': 2, 'text': 'PRIVATE_TRANSCRIPT'}]}


def test_request_logs_are_useful_and_do_not_leak_inputs(caplog):
    caplog.set_level(logging.INFO, logger='uvicorn.error.asr')
    app = create_app(Settings(_env_file=None, asr_api_key=KEY), FakeEngine())
    with TestClient(app) as client:
        response = client.post('/v1/audio/transcriptions',
                               headers={'Authorization': 'Bearer ' + KEY},
                               data={'model': 'whisper-1', 'prompt': 'PRIVATE_PROMPT'},
                               files={'file': ('PRIVATE_FILENAME.mp3', b'PRIVATE_AUDIO')})
        assert response.status_code == 200
        assert app.state.active_task is None
    assert '开始处理' in caplog.text and '文件接收完成' in caplog.text and '任务完成' in caplog.text
    for private in [KEY, 'PRIVATE_FILENAME', 'PRIVATE_PROMPT', 'PRIVATE_AUDIO', 'PRIVATE_TRANSCRIPT']:
        assert private not in caplog.text
    assert current_task.get() is None


def test_download_failure_logs_status_but_not_signed_url(caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger='uvicorn.error.asr')

    def fail(*args):
        raise HTTPException(502, 'PRIVATE_DOWNLOAD_ERROR')

    monkeypatch.setattr('app.main.download_audio', fail)
    app = create_app(Settings(_env_file=None, asr_api_key=KEY), FakeEngine())
    with TestClient(app) as client:
        response = client.post('/v1/audio/transcriptions',
                               headers={'Authorization': 'Bearer ' + KEY},
                               data={'model': 'whisper-1', 'url': 'https://audio.example.com/PRIVATE_PATH?Signature=PRIVATE_SIGNATURE'})
        assert response.status_code == 502
        assert app.state.active_task is None
        assert app.state.inference_slot.acquire(blocking=False)
        app.state.inference_slot.release()
    assert '开始下载' in caplog.text and '任务失败' in caplog.text and 'HTTP=502' in caplog.text
    for private in ['PRIVATE_PATH', 'PRIVATE_SIGNATURE', 'PRIVATE_DOWNLOAD_ERROR']:
        assert private not in caplog.text


def test_model_segment_progress_and_context_cleanup(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger='uvicorn.error.asr')
    package = ModuleType('faster_whisper')
    vad = ModuleType('faster_whisper.vad')
    vad.VadOptions = lambda **kwargs: kwargs
    vad.get_speech_timestamps = lambda *args: [{'start': 0, 'end': 16000}, {'start': 16000, 'end': 32000}]
    monkeypatch.setitem(sys.modules, 'faster_whisper', package)
    monkeypatch.setitem(sys.modules, 'faster_whisper.vad', vad)
    engine = QwenTranscriber(Settings(_env_file=None, asr_api_key=KEY))
    monkeypatch.setattr(engine, 'decode', lambda path: np.zeros(32000, dtype=np.float32))
    engine.model = SimpleNamespace(transcribe=lambda **kwargs: [SimpleNamespace(language='Chinese', text='PRIVATE_TEXT')])
    task = TaskProgress()
    task.start()
    try:
        result = engine.transcribe('PRIVATE_PATH', prompt='PRIVATE_PROMPT')
    finally:
        task.close()
    assert len(result['segments']) == 2
    assert '录音时长=2.0秒' in caplog.text and '第1/2段' in caplog.text and '第2/2段' in caplog.text
    assert '分段进度=100.0%' in caplog.text and task.id in caplog.text
    assert 'PRIVATE_' not in caplog.text
    assert current_task.get() is None and not task.thread.is_alive()


def test_heartbeat_reports_active_phase_without_waiting(caplog):
    caplog.set_level(logging.INFO, logger='uvicorn.error.asr')
    task = TaskProgress()
    task.phase = '识别第2/4段'
    calls = iter([False, True])
    task.stopped = SimpleNamespace(wait=lambda seconds: next(calls))
    task.heartbeat()
    assert '仍在处理' in caplog.text and '识别第2/4段' in caplog.text and task.id in caplog.text


@pytest.mark.parametrize('eos,pad,expected', [(151645, None, 151645), ([151645, 151643], None, 151645), (151645, 7, 7)])
def test_padding_uses_same_default_as_transformers(monkeypatch, eos, pad, expected):
    generation = SimpleNamespace(eos_token_id=eos, pad_token_id=pad)
    wrapped = SimpleNamespace(model=SimpleNamespace(generation_config=generation))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32'))
    monkeypatch.setitem(sys.modules, 'qwen_asr', SimpleNamespace(
        Qwen3ASRModel=SimpleNamespace(from_pretrained=lambda *a, **k: wrapped)))
    engine = QwenTranscriber(Settings(_env_file=None, asr_api_key=KEY))
    engine.load()
    assert generation.pad_token_id == expected

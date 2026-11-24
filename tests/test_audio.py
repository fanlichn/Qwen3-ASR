from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from app.config import Settings
from app.qwen_asr import AudioTooLong, InvalidAudio, QwenTranscriber, SAMPLE_RATE


@pytest.fixture
def engine():
    return QwenTranscriber(Settings(_env_file=None, asr_api_key='test-only-local-key-12345678901234567890',
                                    asr_max_audio_seconds=2))


def test_decode_bounded_command(engine, monkeypatch):
    observed = {}

    def run(command, **kwargs):
        observed.update(command=command, **kwargs)
        return SimpleNamespace(returncode=0, stdout=np.zeros(SAMPLE_RATE, dtype=np.float32).tobytes())

    monkeypatch.setattr(subprocess, 'run', run)
    assert len(engine.decode('local.audio')) == SAMPLE_RATE
    cmd = observed['command']
    assert cmd[cmd.index('-protocol_whitelist') + 1] == 'file,pipe'
    assert cmd[cmd.index('-t') + 1] == '3'
    assert observed['timeout'] == 120


@pytest.mark.parametrize('failure', ['invalid', 'empty', 'timeout', 'long'])
def test_decode_rejects_invalid_or_long_audio(engine, monkeypatch, failure):
    def run(*args, **kwargs):
        if failure == 'timeout':
            raise subprocess.TimeoutExpired('ffmpeg', 120)
        return SimpleNamespace(returncode=1 if failure == 'invalid' else 0,
                               stdout=np.zeros(3 * SAMPLE_RATE, dtype=np.float32).tobytes()
                               if failure == 'long' else b'')

    monkeypatch.setattr(subprocess, 'run', run)
    with pytest.raises(AudioTooLong if failure == 'long' else InvalidAudio):
        engine.decode('local.audio')


def test_compose_only_runs_cpu_api():
    root = Path(__file__).resolve().parents[1]
    compose = yaml.safe_load((root / 'docker-compose.yml').read_text(encoding='utf-8'))
    assert set(compose['services']) == {'api'}
    api = compose['services']['api']
    assert 'deploy' not in api
    assert api['environment']['ASR_DEVICE'] == 'cpu'
    assert 'mysql_data' not in compose.get('volumes', {})
    assert '/readyz' in str(api['healthcheck'])


def test_cpu_is_default_and_cuda_packages_are_not_installed():
    settings = Settings(_env_file=None, asr_api_key='test-only-local-key-12345678901234567890')
    assert settings.asr_device == 'cpu'
    root = Path(__file__).resolve().parents[1]
    lock = (root / 'requirements-linux.lock').read_text(encoding='utf-8')
    assert 'torch==2.6.0+cpu' in lock
    assert 'nvidia-' not in lock

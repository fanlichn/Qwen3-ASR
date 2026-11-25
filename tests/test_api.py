from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.qwen_asr import AudioTooLong, InvalidAudio

KEY = 'test-only-not-a-production-key-1234567890'
AUTH = {'Authorization': f'Bearer {KEY}'}


class FakeTranscriber:
    def __init__(self):
        self.paths = []
        self.calls = []
        self.error = None
        self.loaded = False

    def load(self):
        self.loaded = True

    def transcribe(self, path, **kwargs):
        self.paths.append(Path(path))
        self.calls.append((Path(path).read_bytes(), kwargs))
        if self.error:
            raise self.error
        return {'task': 'transcribe', 'language': 'zh', 'duration': 2.5,
                'text': '你好，测试录音。',
                'segments': [{'id': 0, 'start': 0.2, 'end': 2.5, 'text': '你好，测试录音。'}]}


@pytest.fixture
def api():
    settings = Settings(_env_file=None, asr_api_key=KEY, asr_max_upload_mb=1)
    engine = FakeTranscriber()
    app = create_app(settings, engine)
    with TestClient(app) as client:
        yield client, engine, app


def upload(client, data=None, content=b'fake audio', headers=None, filename='test.mp3'):
    return client.post('/v1/audio/transcriptions', headers=headers if headers is not None else AUTH,
                       files={'file': (filename, content, 'audio/mpeg')},
                       data={'model': 'whisper-1', **(data or {})})


def test_model_discovery_and_auth(api):
    client, engine, _ = api
    assert engine.loaded
    assert client.get('/v1/models').status_code == 401
    assert upload(client, headers={'Authorization': 'Bearer incorrect'}).status_code == 401
    names = [row['id'] for row in client.get('/v1/models', headers=AUTH).json()['data']]
    assert names == ['qwen3-asr-1.7b', 'whisper-1']


def test_health_and_removed_qc_routes(api):
    client, _, app = api
    assert client.get('/healthz').status_code == 200
    assert client.get('/readyz').json() == {'status': 'ready'}
    app.state.ready = False
    assert client.get('/readyz').status_code == 503
    assert upload(client).status_code == 503
    for path in ['/', '/api/recordings', '/login', '/docs']:
        assert client.get(path).status_code == 404


def test_transcription_and_cleanup(api):
    client, engine, _ = api
    response = upload(client, {'language': 'ZH', 'prompt': '术语：顺丰'}, filename='../../escape.mp3')
    assert response.status_code == 200
    assert response.json() == {'text': '你好，测试录音。'}
    assert engine.calls == [(b'fake audio', {'language': 'zh', 'prompt': '术语：顺丰'})]
    assert engine.paths[0].name == 'upload.audio'
    assert not engine.paths[0].exists()
    assert upload(client, {'model': 'qwen3-asr-1.7b'}).status_code == 200


@pytest.mark.parametrize('format', ['text', 'verbose_json', 'srt', 'vtt'])
def test_response_formats(api, format):
    client, _, _ = api
    response = upload(client, {'response_format': format})
    assert response.status_code == 200
    if format == 'verbose_json':
        assert response.json()['segments'][0]['start'] == 0.2
        assert response.json()['duration'] == 2.5
    elif format == 'srt':
        assert '00:00:00,200 --> 00:00:02,500' in response.text
    elif format == 'vtt':
        assert response.text.startswith('WEBVTT\n')
        assert '00:00:00.200 --> 00:00:02.500' in response.text
    else:
        assert response.text == '你好，测试录音。'


@pytest.mark.parametrize('data', [
    {'model': 'wrong'}, {'language': 'xx'}, {'response_format': 'wrong'},
    {'temperature': '0.5'}, {'temperature': 'nan'}, {'temperature': 'invalid'},
    {'stream': 'true'}, {'timestamp_granularities[]': 'word'}, {'prompt': 'a' * 2001},
])
def test_unsupported_options_do_not_infer(api, data):
    client, engine, _ = api
    assert upload(client, data).status_code == 400
    assert not engine.calls


def test_required_fields_and_empty_upload(api):
    client, engine, _ = api
    assert client.post('/v1/audio/transcriptions', headers=AUTH).status_code == 400
    assert upload(client, content=b'').status_code == 400
    assert not engine.calls


def test_url_input_passes_downloaded_audio_to_existing_engine(api, monkeypatch):
    client, engine, _ = api

    def download(url, path, settings):
        assert url == 'https://audio.example.com/a.mp3'
        path.write_bytes(b'downloaded audio')

    monkeypatch.setattr('app.main.download_audio', download)
    response = client.post('/v1/audio/transcriptions', headers=AUTH,
                           data={'model': 'whisper-1', 'url': 'https://audio.example.com/a.mp3'})
    assert response.status_code == 200
    assert engine.calls[0][0] == b'downloaded audio'
    assert not engine.paths[0].exists()


def test_url_and_file_are_mutually_exclusive(api):
    client, engine, _ = api
    assert upload(client, {'url': 'https://audio.example.com/a.mp3'}).status_code == 400
    assert not engine.calls


def test_url_authentication_and_configuration(api):
    client, engine, _ = api
    data = {'model': 'whisper-1', 'url': 'https://audio.example.com/a.mp3'}
    assert client.post('/v1/audio/transcriptions', data=data).status_code == 401
    response = client.post('/v1/audio/transcriptions', headers=AUTH, data=data)
    assert response.status_code == 400
    assert 'configure ASR_URL_ALLOWED_HOSTS' in response.json()['error']['message']
    assert not engine.calls


def test_upload_limits_and_auth_before_parsing(api):
    client, engine, _ = api
    assert upload(client, content=b'a' * (1024 * 1024 + 1)).status_code == 413
    assert upload(client, content=b'a' * (3 * 1024 * 1024)).status_code == 413
    assert upload(client, content=b'a' * (3 * 1024 * 1024), headers={}).status_code == 401
    assert not engine.calls


@pytest.mark.parametrize('error,status', [(InvalidAudio('Invalid audio'), 400),
                                        (AudioTooLong('too long'), 413),
                                        (RuntimeError('private inference detail'), 500)])
def test_failure_cleanup_and_recovery(api, error, status):
    client, engine, _ = api
    engine.error = error
    response = upload(client)
    assert response.status_code == status
    assert 'private inference detail' not in response.text
    assert not engine.paths[-1].exists()
    engine.error = None
    assert upload(client).status_code == 200


def test_busy_gpu_is_retryable(api):
    client, engine, app = api
    app.state.inference_slot.acquire()
    try:
        response = upload(client)
        assert response.status_code == 503
        assert response.headers['retry-after'] == '5'
        assert not engine.calls
    finally:
        app.state.inference_slot.release()
    assert upload(client).status_code == 200


def test_chunked_request_size_limit_without_content_length(api):
    client, engine, _ = api

    def body():
        yield b'--boundary\r\nContent-Disposition: form-data; name="file"; filename="large.mp3"\r\n'
        yield b'Content-Type: audio/mpeg\r\n\r\n'
        for _ in range(4):
            yield b'a' * (1024 * 1024)
        yield b'\r\n--boundary--\r\n'

    response = client.post('/v1/audio/transcriptions', headers={**AUTH,
                           'Content-Type': 'multipart/form-data; boundary=boundary'}, content=body())
    assert response.status_code == 413
    assert not engine.calls


def test_custom_model_name():
    settings = Settings(_env_file=None, asr_api_key=KEY, asr_model_name='my-asr')
    with TestClient(create_app(settings, FakeTranscriber())) as client:
        assert upload(client, {'model': 'my-asr'}).status_code == 200
        assert upload(client, {'model': 'qwen3-asr-1.7b'}).status_code == 400


def test_invalid_secret_not_exposed():
    value = 'private-invalid-key'
    with pytest.raises(ValueError) as caught:
        Settings(_env_file=None, asr_api_key=value)
    assert value not in str(caught.value)


def test_startup_failure_is_not_ready():
    engine = FakeTranscriber()

    def fail():
        raise RuntimeError('private model location or token')

    engine.load = fail
    app = create_app(Settings(_env_file=None, asr_api_key=KEY), engine)
    with pytest.raises(RuntimeError, match='ASR startup failed') as caught:
        with TestClient(app):
            pass
    assert 'private model location or token' not in str(caught.value)
    assert app.state.ready is False

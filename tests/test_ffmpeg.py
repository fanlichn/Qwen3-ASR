"""Real FFmpeg decoding tests with synthetic audio, without a GPU or model download."""

import subprocess
import wave

import imageio_ffmpeg
import numpy as np
import pytest

from app.config import Settings
from app.qwen_asr import AudioTooLong, InvalidAudio, QwenTranscriber


@pytest.fixture
def real_decoder(monkeypatch):
    binary = imageio_ffmpeg.get_ffmpeg_exe()
    original_run = subprocess.run

    def run(command, **kwargs):
        return original_run([binary, *command[1:]], **kwargs)

    monkeypatch.setattr(subprocess, 'run', run)
    return QwenTranscriber(Settings(_env_file=None,
                           asr_api_key='test-only-local-key-12345678901234567890',
                           asr_max_audio_seconds=2))


def write_wav(path, duration):
    rate = 16000
    samples = (np.sin(np.arange(int(rate * duration)) / rate * 440 * 2 * np.pi) * 8000).astype('<i2')
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(samples.tobytes())


@pytest.mark.parametrize('extension', ['wav', 'mp3', 'm4a', 'flac', 'ogg'])
def test_real_audio_formats(real_decoder, tmp_path, extension):
    source = tmp_path / 'source.wav'
    write_wav(source, 1)
    encoded = tmp_path / f'audio.{extension}'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), str(encoded)], check=True)
    # Production stores every upload using a fixed extension, not the user filename.
    upload = tmp_path / 'upload.audio'
    encoded.rename(upload)
    samples = real_decoder.decode(str(upload))
    assert 15000 <= len(samples) <= 18000
    assert np.max(np.abs(samples)) > 0.1


def test_real_overlong_audio_is_rejected(real_decoder, tmp_path):
    source = tmp_path / 'overlong.wav'
    write_wav(source, 4)
    with pytest.raises(AudioTooLong):
        real_decoder.decode(str(source))


def test_real_corrupt_audio_is_rejected(real_decoder, tmp_path):
    source = tmp_path / 'broken.audio'
    source.write_bytes(b'this is not a recording')
    with pytest.raises(InvalidAudio):
        real_decoder.decode(str(source))

"""Local Qwen inference; heavy dependencies load only at service startup."""

from collections import Counter
import subprocess

from .config import Settings

SAMPLE_RATE = 16000
LANGUAGE_CODES = {
    'Arabic': 'ar', 'Cantonese': 'yue', 'Chinese': 'zh', 'Czech': 'cs', 'Danish': 'da',
    'Dutch': 'nl', 'English': 'en', 'Filipino': 'fil', 'Finnish': 'fi', 'French': 'fr',
    'German': 'de', 'Greek': 'el', 'Hindi': 'hi', 'Hungarian': 'hu', 'Indonesian': 'id',
    'Italian': 'it', 'Japanese': 'ja', 'Korean': 'ko', 'Macedonian': 'mk', 'Malay': 'ms',
    'Persian': 'fa', 'Polish': 'pl', 'Portuguese': 'pt', 'Romanian': 'ro', 'Russian': 'ru',
    'Spanish': 'es', 'Swedish': 'sv', 'Thai': 'th', 'Turkish': 'tr', 'Vietnamese': 'vi',
}
LANGUAGE_NAMES = {code: name for name, code in LANGUAGE_CODES.items()}


class InvalidAudio(ValueError):
    pass


class AudioTooLong(ValueError):
    pass


class QwenTranscriber:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None

    def load(self):
        import torch
        from qwen_asr import Qwen3ASRModel

        if self.settings.asr_device.startswith('cuda:'):
            index = int(self.settings.asr_device.split(':')[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError('Configured CUDA device is unavailable')
            dtype = torch.float16
        else:
            dtype = torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            self.settings.asr_model_path, dtype=dtype, device_map=self.settings.asr_device,
            attn_implementation='sdpa', max_inference_batch_size=1, max_new_tokens=512,
        )

    def decode(self, path: str):
        import numpy as np

        # Limit decoded memory and prevent network requests from playlist input.
        try:
            result = subprocess.run(
                ['ffmpeg', '-nostdin', '-v', 'error', '-protocol_whitelist', 'file,pipe',
                 '-i', path, '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE),
                 '-t', str(self.settings.asr_max_audio_seconds + 1), '-f', 'f32le', 'pipe:1'],
                capture_output=True, timeout=self.settings.asr_decode_timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise InvalidAudio('Audio decoding timed out') from exc
        if result.returncode or not result.stdout:
            raise InvalidAudio('The uploaded file could not be decoded as audio')
        audio = np.frombuffer(result.stdout, dtype=np.float32).copy()
        if len(audio) > self.settings.asr_max_audio_seconds * SAMPLE_RATE:
            raise AudioTooLong('Audio exceeds ASR_MAX_AUDIO_SECONDS')
        return audio

    def transcribe(self, path: str, *, language: str | None = None, prompt: str = '') -> dict:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        audio = self.decode(path)
        speech = get_speech_timestamps(
            audio, VadOptions(max_speech_duration_s=self.settings.asr_chunk_seconds),
        )
        rows = []
        language_samples = Counter()
        for segment in speech:
            start, end = segment['start'], segment['end']
            result = self.model.transcribe(
                audio=(audio[start:end], SAMPLE_RATE),
                language=LANGUAGE_NAMES.get(language), context=prompt,
            )[0]
            text = result.text.strip()
            if text:
                code = LANGUAGE_CODES.get(result.language)
                if code:
                    language_samples[code] += end - start
                rows.append({'id': len(rows), 'start': round(start / SAMPLE_RATE, 2),
                             'end': round(end / SAMPLE_RATE, 2), 'text': text})
        detected = language_samples.most_common(1)[0][0] if language_samples else 'und'
        return {'task': 'transcribe', 'language': language or detected,
                'duration': round(len(audio) / SAMPLE_RATE, 3),
                'text': '\n'.join(row['text'] for row in rows), 'segments': rows}

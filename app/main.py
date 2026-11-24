import hmac
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import Settings
from .qwen_asr import AudioTooLong, InvalidAudio, LANGUAGE_NAMES, QwenTranscriber

logger = logging.getLogger(__name__)


def error_body(message: str, status: int):
    return {'error': {'message': message, 'type': 'authentication_error' if status == 401
                     else 'invalid_request_error' if status < 500 else 'server_error',
                     'param': None, 'code': str(status)}}


class RequestGuard:
    """Authenticate and cap the body before multipart parsing can spool files."""

    def __init__(self, app, settings: Settings):
        self.app = app
        self.key = settings.asr_api_key.get_secret_value().encode('utf-8')
        self.max_bytes = settings.asr_max_upload_mb * 1024 * 1024 + 1024 * 1024

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or not scope['path'].startswith('/v1/'):
            return await self.app(scope, receive, send)
        headers = dict(scope['headers'])
        auth = headers.get(b'authorization', b'').split(b' ', 1)
        if len(auth) != 2 or auth[0].lower() != b'bearer' or not hmac.compare_digest(auth[1], self.key):
            response = JSONResponse(error_body('Invalid or missing API key', 401), 401,
                                    headers={'WWW-Authenticate': 'Bearer'})
            return await response(scope, receive, send)
        try:
            length = int(headers.get(b'content-length', b'0'))
        except ValueError:
            return await JSONResponse(error_body('Invalid Content-Length', 400), 400)(scope, receive, send)
        if length > self.max_bytes:
            return await JSONResponse(error_body('Request body is too large', 413), 413)(scope, receive, send)
        consumed = 0

        async def limited_receive():
            nonlocal consumed
            message = await receive()
            if message['type'] == 'http.request':
                consumed += len(message.get('body', b''))
                if consumed > self.max_bytes:
                    raise StarletteHTTPException(413, 'Request body is too large')
            return message

        await self.app(scope, limited_receive, send)


def subtitle_text(segments: list[dict], webvtt: bool) -> str:
    def timestamp(seconds):
        millis = round(seconds * 1000)
        hours, millis = divmod(millis, 3600000)
        minutes, millis = divmod(millis, 60000)
        secs, millis = divmod(millis, 1000)
        return f'{hours:02}:{minutes:02}:{secs:02}{"." if webvtt else ","}{millis:03}'

    rows = ['WEBVTT\n'] if webvtt else []
    for index, row in enumerate(segments, 1):
        rows.append(f'{index}\n{timestamp(row["start"])} --> {timestamp(row["end"])}\n{row["text"]}\n')
    return '\n'.join(rows)


def create_app(settings: Settings | None = None, transcriber=None) -> FastAPI:
    settings = settings or Settings()
    engine = transcriber if transcriber is not None else QwenTranscriber(settings)

    @asynccontextmanager
    async def lifespan(application):
        application.state.ready = False
        try:
            engine.load()
        except Exception as exc:
            logger.error('ASR startup failed (%s)', type(exc).__name__)
            raise RuntimeError('ASR startup failed; check GPU availability and model files') from None
        application.state.ready = True
        yield
        application.state.ready = False

    application = FastAPI(title='Speech to Text API', version='1.0.0', lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    application.state.ready = False
    application.state.inference_slot = BoundedSemaphore(1)
    application.add_middleware(RequestGuard, settings=settings)

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse(error_body(str(exc.detail), exc.status_code), exc.status_code, headers=exc.headers)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(error_body('Invalid form fields; file and model are required', 400), 400)

    @application.get('/healthz')
    def health():
        return {'status': 'ok'}

    @application.get('/readyz')
    def readiness():
        return JSONResponse({'status': 'ready' if application.state.ready else 'not_ready'},
                            200 if application.state.ready else 503)

    @application.get('/v1/models')
    def models():
        names = list(dict.fromkeys([settings.asr_model_name, 'whisper-1']))
        return {'object': 'list', 'data': [
            {'id': name, 'object': 'model', 'created': 0, 'owned_by': 'local'} for name in names
        ]}

    @application.post('/v1/audio/transcriptions')
    def transcription(
        file: Annotated[UploadFile, File()],
        model: Annotated[str, Form()],
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str, Form()] = '',
        response_format: Annotated[str, Form()] = 'json',
        temperature: Annotated[float, Form()] = 0,
        stream: Annotated[bool, Form()] = False,
        timestamp_granularities: Annotated[list[str] | None, Form(alias='timestamp_granularities[]')] = None,
    ):
        if not application.state.ready:
            raise HTTPException(503, 'The speech recognition model is not ready')
        if model not in {settings.asr_model_name, 'whisper-1'}:
            raise HTTPException(400, 'Unknown model; see GET /v1/models')
        if response_format not in {'json', 'text', 'verbose_json', 'srt', 'vtt'}:
            raise HTTPException(400, 'Unsupported response_format')
        if language is not None:
            language = language.lower()
            if language not in LANGUAGE_NAMES:
                raise HTTPException(400, 'Unsupported language code; omit it for automatic detection')
        if len(prompt) > 2000:
            raise HTTPException(400, 'prompt must not exceed 2000 characters')
        if temperature != 0 or stream:
            raise HTTPException(400, 'Only temperature=0 and stream=false are supported')
        if timestamp_granularities and any(item != 'segment' for item in timestamp_granularities):
            raise HTTPException(400, 'Only segment timestamps are supported')
        if not application.state.inference_slot.acquire(blocking=False):
            raise HTTPException(503, 'The GPU is busy; retry later', headers={'Retry-After': '5'})
        try:
            with TemporaryDirectory(prefix='asr-') as directory:
                audio_path = Path(directory) / 'upload.audio'
                count = 0
                with audio_path.open('wb') as destination:
                    while chunk := file.file.read(1024 * 1024):
                        count += len(chunk)
                        if count > settings.asr_max_upload_mb * 1024 * 1024:
                            raise HTTPException(413, 'Audio file exceeds ASR_MAX_UPLOAD_MB')
                        destination.write(chunk)
                if not count:
                    raise HTTPException(400, 'The audio file is empty')
                result = engine.transcribe(str(audio_path), language=language, prompt=prompt)
        except AudioTooLong:
            raise HTTPException(413, 'Audio exceeds ASR_MAX_AUDIO_SECONDS') from None
        except InvalidAudio as exc:
            raise HTTPException(400, str(exc)) from None
        except HTTPException:
            raise
        except Exception as exc:
            logger.error('ASR inference failed (%s)', type(exc).__name__)
            raise HTTPException(500, 'Speech recognition failed; check service GPU and model status') from None
        finally:
            application.state.inference_slot.release()
            file.file.close()
        if response_format == 'json':
            return {'text': result['text']}
        if response_format == 'text':
            return PlainTextResponse(result['text'])
        if response_format in {'srt', 'vtt'}:
            return PlainTextResponse(subtitle_text(result['segments'], response_format == 'vtt'),
                                     media_type='text/vtt' if response_format == 'vtt' else 'application/x-subrip')
        return result

    return application

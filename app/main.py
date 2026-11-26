import hmac
import logging
import time
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
from .remote_audio import download_audio
from .progress import TaskProgress

logger = logging.getLogger('uvicorn.error.asr')


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
        load_started = time.monotonic()
        logger.info('服务启动 | 开始加载语音模型 | 设备=%s', settings.asr_device)
        try:
            engine.load()
        except Exception as exc:
            logger.error('ASR startup failed (%s)', type(exc).__name__)
            raise RuntimeError('ASR startup failed; check device availability and model files') from None
        application.state.ready = True
        logger.info('服务就绪 | 模型加载耗时=%.1f秒 | 同时处理任务数=1', time.monotonic() - load_started)
        yield
        application.state.ready = False
        logger.info('服务已关闭')

    application = FastAPI(title='Speech to Text API', version='1.0.0', lifespan=lifespan,
                          docs_url=None, redoc_url=None, openapi_url=None)
    application.state.ready = False
    application.state.inference_slot = BoundedSemaphore(1)
    application.state.active_task = None
    application.add_middleware(RequestGuard, settings=settings)

    @application.exception_handler(StarletteHTTPException)
    async def http_error(request, exc):
        return JSONResponse(error_body(str(exc.detail), exc.status_code), exc.status_code, headers=exc.headers)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(error_body('Invalid form fields; model and either file or url are required', 400), 400)

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
        model: Annotated[str, Form()],
        file: Annotated[UploadFile | None, File()] = None,
        url: Annotated[str | None, Form()] = None,
        language: Annotated[str | None, Form()] = None,
        prompt: Annotated[str, Form()] = '',
        response_format: Annotated[str, Form()] = 'json',
        temperature: Annotated[float, Form()] = 0,
        stream: Annotated[bool, Form()] = False,
        timestamp_granularities: Annotated[list[str] | None, Form(alias='timestamp_granularities[]')] = None,
    ):
        if not application.state.ready:
            raise HTTPException(503, 'The speech recognition model is not ready')
        if (file is not None) == bool(url):
            raise HTTPException(400, 'Provide exactly one of file or url')
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
        task = TaskProgress()
        if not application.state.inference_slot.acquire(blocking=False):
            active = application.state.active_task
            task.emit('请求被拒绝 | HTTP=503 | 服务忙碌 | 正在处理任务=%s', active.id if active else '-')
            raise HTTPException(503, 'Speech recognition is busy; retry later', headers={'Retry-After': '5'})
        application.state.active_task = task
        try:
            task.start()
            task.emit('开始处理 | 输入来源=%s', '远程链接' if url else '上传文件')
            with TemporaryDirectory(prefix='asr-') as directory:
                audio_path = Path(directory) / 'upload.audio'
                input_started = time.monotonic()
                if url:
                    task.emit('开始下载远程录音', phase='下载录音')
                    download_audio(url, audio_path, settings)
                    task.emit('下载完成 | 大小=%.2fMiB | 下载耗时=%.1f秒',
                              audio_path.stat().st_size / 1024**2, time.monotonic() - input_started)
                else:
                    task.emit('正在接收上传文件', phase='接收文件')
                    count = 0
                    with audio_path.open('wb') as destination:
                        while chunk := file.file.read(1024 * 1024):
                            count += len(chunk)
                            if count > settings.asr_max_upload_mb * 1024 * 1024:
                                raise HTTPException(413, 'Audio file exceeds ASR_MAX_UPLOAD_MB')
                            destination.write(chunk)
                    if not count:
                        raise HTTPException(400, 'The audio file is empty')
                    task.emit('文件接收完成 | 大小=%.2fMiB', count / 1024**2)
                result = engine.transcribe(str(audio_path), language=language, prompt=prompt)
                task.emit('任务完成 | 录音时长=%.1f秒 | 输出段数=%d | 文字字符数=%d | 总耗时=%.1f秒',
                          result['duration'], len(result['segments']), len(result['text']),
                          time.monotonic() - task.started, phase='完成')
        except AudioTooLong:
            task.emit('任务失败 | HTTP=413 | 原因=录音时长超限 | 已耗时=%.1f秒', time.monotonic() - task.started)
            raise HTTPException(413, 'Audio exceeds ASR_MAX_AUDIO_SECONDS') from None
        except InvalidAudio as exc:
            task.emit('任务失败 | HTTP=400 | 原因=音频解码失败 | 已耗时=%.1f秒', time.monotonic() - task.started)
            raise HTTPException(400, str(exc)) from None
        except HTTPException as exc:
            task.emit('任务失败 | HTTP=%d | 阶段=%s | 已耗时=%.1f秒',
                      exc.status_code, task.phase, time.monotonic() - task.started)
            raise
        except Exception as exc:
            logger.error('任务=%s 任务失败 | 异常类型=%s | 阶段=%s | 已耗时=%.1f秒',
                         task.id, type(exc).__name__, task.phase, time.monotonic() - task.started)
            raise HTTPException(500, 'Speech recognition failed; check service device and model status') from None
        finally:
            task.close()
            application.state.active_task = None
            application.state.inference_slot.release()
            if file is not None:
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

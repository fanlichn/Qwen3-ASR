"""Privacy-safe task progress, using Uvicorn's existing INFO log handler."""

from contextvars import ContextVar
import logging
from threading import Event, Thread
import time
from uuid import uuid4

logger = logging.getLogger('uvicorn.error.asr')
current_task = ContextVar('asr_log_task', default=None)


class DefaultPaddingFilter(logging.Filter):
    def filter(self, record):
        return not record.getMessage().startswith('Setting `pad_token_id` to `eos_token_id`:')


def configure_model_logging():
    model_logger = logging.getLogger('transformers.generation.utils')
    if not any(isinstance(item, DefaultPaddingFilter) for item in model_logger.filters):
        model_logger.addFilter(DefaultPaddingFilter())


class TaskProgress:
    def __init__(self):
        self.id = uuid4().hex[:12]
        self.started = time.monotonic()
        self.phase = '准备输入'
        self.stopped = Event()
        self.thread = None
        self.token = None

    def emit(self, message, *args, phase=None):
        if phase is not None:
            self.phase = phase
        logger.info('任务=%s ' + message, self.id, *args)

    def heartbeat(self):
        while not self.stopped.wait(30):
            self.emit('仍在处理 | 阶段=%s | 已耗时=%.1f秒',
                      self.phase, time.monotonic() - self.started)

    def start(self):
        self.token = current_task.set(self)
        self.thread = Thread(target=self.heartbeat, name='asr-progress', daemon=True)
        self.thread.start()

    def close(self):
        self.stopped.set()
        if self.thread is not None and self.thread.ident is not None:
            self.thread.join(timeout=1)
        if self.token is not None:
            current_task.reset(self.token)
            self.token = None


def progress(message, *args, phase=None):
    task = current_task.get()
    if task is not None:
        task.emit(message, *args, phase=phase)

"""Production entrypoint; tests can import app.main without downloading model weights."""

from .main import create_app

app = create_app()

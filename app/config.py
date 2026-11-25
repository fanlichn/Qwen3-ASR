from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore', hide_input_in_errors=True)

    asr_api_key: SecretStr
    asr_model_path: str = 'Qwen/Qwen3-ASR-1.7B'
    asr_model_name: str = 'qwen3-asr-1.7b'
    asr_device: str = 'cpu'
    asr_max_upload_mb: int = Field(default=25, ge=1, le=200)
    asr_max_audio_seconds: int = Field(default=1800, ge=1, le=7200)
    asr_chunk_seconds: int = Field(default=25, ge=5, le=25)
    asr_decode_timeout_seconds: int = Field(default=120, ge=1, le=600)
    asr_url_allowed_hosts: str = ''
    asr_url_timeout_seconds: int = Field(default=60, ge=1, le=300)

    @field_validator('asr_api_key')
    @classmethod
    def validate_key(cls, value: SecretStr) -> SecretStr:
        raw = value.get_secret_value()
        if len(raw) < 32 or raw != raw.strip() or raw.lower().startswith(('replace', 'change', 'your-')):
            raise ValueError('ASR_API_KEY must be a newly generated key with at least 32 characters')
        return value

    @field_validator('asr_model_name')
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError('ASR_MODEL_NAME cannot be empty or contain surrounding whitespace')
        return value

    @field_validator('asr_device')
    @classmethod
    def validate_device(cls, value: str) -> str:
        if value != 'cpu' and not (value.startswith('cuda:') and value[5:].isdigit()):
            raise ValueError('ASR_DEVICE must be cpu or cuda:N')
        return value

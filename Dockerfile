ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
ARG REQUIREMENTS_LOCK=requirements-linux.lock

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/huggingface \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ffmpeg libgomp1 libsndfile1 sox ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv/app
COPY requirements*.txt requirements-*.lock ./
RUN python3 -m pip install --upgrade pip \
    && python3 -m pip install torch==2.6.0 --index-url "${TORCH_INDEX_URL}" \
    && python3 -m pip install -r "${REQUIREMENTS_LOCK}" \
    && python3 -m pip check

COPY app ./app
RUN useradd --uid 10001 --create-home asr \
    && mkdir -p /data/huggingface && chown -R asr:asr /data/huggingface /srv/app
USER asr

EXPOSE 8080
# One process per GPU: multiple workers would load multiple copies of the model.
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]

# 录音转文字 API

本项目已精简为独立语音识别服务：上传音频，返回文字。使用本地 GPU 运行 **Qwen3-ASR-1.7B**，可作为 New API 的语音识别上游。

调用关系：`业务程序 → New API → 本服务 → GPU 语音识别 → 文字`。

原 Web 后台、OSS 采集、Worker、质检评分、DeepSeek 分析及 MySQL 依赖已移出当前代码；已有数据库、录音、Docker 数据卷不做任何删除。旧版可以从 Git 历史或修改前备份找回。

## 1. Ubuntu 22.04 GPU 服务器部署

需要 NVIDIA 显卡驱动、Docker Engine、Docker Compose v2，以及 NVIDIA Container Toolkit。Ubuntu 系统版本本身不能保证 GPU 容器已经配置好。先执行：

```bash
nvidia-smi
docker --version
docker compose version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

最后一条必须能显示显卡。CUDA 12.4 / PyTorch 2.6 的驱动兼容要求以 [NVIDIA 官方说明](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html) 为准。Docker / GPU 容器环境没有配置时，参考 [Docker Ubuntu 安装说明](https://docs.docker.com/engine/install/ubuntu/) 和 [NVIDIA Container Toolkit 安装说明](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)。无需在宿主机 Conda 环境中安装 Python 依赖。

把当前版本代码上传到服务器，例如 `/data/ai-collection-qc`。首次部署：

```bash
cd /data/ai-collection-qc
# 仅当没有 .env 时复制模板，保留已有配置文件。
test -f .env || cp .env.example .env
chmod 600 .env
nano .env
```

在另一个终端执行 `openssl rand -hex 32` 生成随机密钥，把生成的内容填到 `.env` 的 `ASR_API_KEY=` 后。这个密钥是你自己给本服务设置的访问密码；后续填进 New API 渠道。无需购买千问 API Key。若沿用旧 `.env`，需要补齐 `.env.example` 中的 `ASR_` 配置；旧 OSS / 数据库变量会被忽略。

默认监听 `0.0.0.0:8080`；可在 `.env` 修改端口和绑定地址。远程访问需要云服务器安全组开放对应端口，建议仅允许 New API 服务器访问；公网使用时通过 HTTPS 反向代理。运行：

```bash
docker compose -f docker-compose.yml up -d --build api
docker compose -f docker-compose.yml logs --tail=100 -f api
```

显式指定 `-f` 可避免旧版 `docker-compose.override.yml` 自动覆盖配置。第一次构建会下载 PyTorch 等依赖；第一次启动会下载模型到 `hf_cache` 卷，耗时取决于网络。服务在模型加载完成后才开始接收请求。若 Hugging Face 无法访问，可通过 `.env` 的 `HF_ENDPOINT` 使用你信任的兼容模型仓库，或预先下载模型并挂载到容器，将 `ASR_MODEL_PATH` 指向该目录。

新 Compose 只启动 `api`，不会自动停止旧版已运行的 Worker / MySQL。若复用旧部署，请先检查旧容器，避免旧 Worker 继续占用 GPU；不要删除原来的数据库卷。当前目录下的备份和 `.env` 不会放进镜像。

检查：

```bash
curl -fsS http://127.0.0.1:8080/readyz
docker compose -f docker-compose.yml exec api python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU:', torch.cuda.get_device_name(0))"
```

就绪接口应返回 `{"status":"ready"}`。`/healthz` 仅用于进程存活检测。模型启动失败时容器退出并按重启策略重试，不会伪报已就绪。

## 2. 先直接测试服务

下面的命令交互读取密钥，不在命令历史里写明文。将录音路径替换为你的实际文件：

```bash
read -rsp 'ASR service key: ' ASR_TEST_KEY; echo
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer ${ASR_TEST_KEY}" \
  -F 'model=whisper-1' \
  -F 'language=zh' \
  -F 'file=@/data/test.mp3'
unset ASR_TEST_KEY
```

返回示例（仅展示格式，不代表实际识别结果）：

```json
{"text":"你好，这是一段录音。"}
```

`whisper-1` 是为网关兼容提供的模型别名，实际运行的仍是 Qwen3-ASR；也支持 `model=qwen3-asr-1.7b`。MP3、WAV、M4A、FLAC、OGG 等由 FFmpeg 解码，不根据扩展名判断。接口只接受上传文件，不接受远程录音 URL。

## 3. 接入 New API

New API 的官方路由支持 [POST /v1/audio/transcriptions](https://github.com/QuantumNous/new-api/blob/main/router/relay-router.go)。在其管理后台新增渠道（具体字段名称随版本可能变化）：

| 字段 | 填写内容 |
| --- | --- |
| 渠道类型 | OpenAI / OpenAI 兼容 |
| 名称 | 录音转文字 |
| 代理地址 / Base URL | `http://本服务IP:8080`，不要附加 `/v1` |
| 密钥 | 本服务 `.env` 的 `ASR_API_KEY` |
| 模型 | 先填 `whisper-1`，无需模型映射 |
| 分组 | 选择业务程序令牌有权使用的分组 |

New API 和本服务分别在 Docker 容器时，`127.0.0.1` 指各自容器，不能代表对方。使用可达的内网 IP，或将两者接到同一个 Docker 网络后使用服务名。New API 中还需要按你的版本设置模型计费 / 倍率及令牌模型权限。通道的默认聊天测试可能请求 `/v1/chat/completions`，本服务不提供这个接口；请使用下方音频请求验证。

业务程序使用 **New API 令牌**，不使用本服务密钥：

```bash
read -rsp 'New API token: ' NEWAPI_TEST_TOKEN; echo
curl -sS http://你的NewAPI地址/v1/audio/transcriptions \
  -H "Authorization: Bearer ${NEWAPI_TEST_TOKEN}" \
  -F 'model=whisper-1' \
  -F 'language=zh' \
  -F 'file=@/data/test.mp3'
unset NEWAPI_TEST_TOKEN
```

New API、前置反向代理和调用程序都要允许音频上传大小，并将请求超时设为可覆盖实际识别耗时的值，例如 600 秒起。长录音可能需要更长超时，应使用实际录音测量。

## 4. API 约定

`GET /v1/models` 和 `POST /v1/audio/transcriptions` 均要求 `Authorization: Bearer <ASR_API_KEY>`。健康接口无需鉴权。无后台页面、注册登录、任务数据库。

| 参数 | 说明 |
| --- | --- |
| `file` | 必填，multipart 音频文件 |
| `model` | 必填，`whisper-1` 或 `ASR_MODEL_NAME` |
| `language` | 可选，`zh`、`en` 等；不传则自动检测 |
| `prompt` | 可选，最多 2000 字符，作为术语 / 上下文提示送入识别模型 |
| `response_format` | `json`（默认）、`text`、`verbose_json`、`srt`、`vtt` |
| `temperature` | 仅支持 `0` |
| `stream` | 仅支持 `false`，当前为同步文件转录 |
| `timestamp_granularities[]` | 仅支持 `segment`；不支持逐字时间戳 |

`verbose_json` 返回 `task / language / duration / text / segments`。分段时间来自 VAD 检测的语音区间，是近似时间，不是逐字对齐。静音录音可返回空文字。默认最大上传 25 MiB，最长 1800 秒；超限返回 `413`，不会静默截断。格式错误 / 无法解码返回 `400`，鉴权失败 `401`。

单进程每次处理一条录音，避免多个任务竞争 GPU 显存；忙时返回 `503`，附 `Retry-After: 5`，调用方稍后重试。不包含持久任务队列；上传音频只在临时目录中处理，请求结束后清理。录音正文与密钥不写入应用日志。当前不提供说话人分离、业务质检或语音合成。

## 5. 代码与本地接口测试

| 文件 | 作用 |
| --- | --- |
| `app/server.py` | 生产启动入口 |
| `app/main.py` | 音频接口、鉴权、大小限制、并发保护、输出格式 |
| `app/qwen_asr.py` | FFmpeg 解码、VAD 分段、本地 Qwen 推理 |
| `app/config.py` | `.env` 配置和校验 |
| `Dockerfile` / `docker-compose.yml` | Ubuntu 22.04 CUDA 镜像及单服务 GPU 部署 |
| `requirements-linux.lock` | 针对 Linux x86_64 / Python 3.10 解析的固定依赖版本 |
| `tests/` | 无 GPU 的接口及音频处理回归测试 |

```bash
python -m pip install -c requirements-linux.lock -r requirements-dev.txt
python -m pytest -q
```

接口测试使用替身识别器验证 HTTP 协议；不下载模型，也不能代替服务器上实际 GPU 识别和 New API 端到端测试。推理依赖和模型接口依据 [Qwen3-ASR 官方项目](https://github.com/QwenLM/Qwen3-ASR)。

当前本地验证记录见 [VALIDATION.md](VALIDATION.md)。

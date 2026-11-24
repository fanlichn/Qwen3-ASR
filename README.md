# 录音转文字 API

上传音频，返回文字。使用本地 **Qwen3-ASR-1.7B**，默认在 **CPU** 上运行，可作为 New API 的语音识别上游。无需 NVIDIA GPU，也无需购买千问 / DeepSeek / OpenAI API Key。

调用关系：`业务程序 → New API → 本服务 → 语音识别 → 文字`。

已配置域名访问方案：`https://asr.lakdriya.com`。Nginx 配置文件及 Cloudflare 无源站证书设置见 [deploy/NGINX.md](deploy/NGINX.md)。

当前只包含录音转文字，不包含 Web 业务后台、OSS 采集、Worker、质检评分和 MySQL。已有数据库、录音和 Docker 数据卷不做删除。

## 1. Ubuntu 22.04 部署

需要 Docker Engine 和 Docker Compose。依赖安装在容器中，不需要修改宿主机 Conda 环境。官方安装说明：[Docker Ubuntu](https://docs.docker.com/engine/install/ubuntu/)。

```bash
cd /data
# 首次部署：
git clone https://github.com/fanlichn/Qwen3-ASR.git
cd Qwen3-ASR
# 后续更新：
git pull --ff-only
```

首次配置（保留已有 `.env`）：

```bash
test -f .env || cp .env.example .env
chmod 600 .env
nano .env
```

执行 `openssl rand -hex 32` 生成随机密钥，填到 `.env` 的 `ASR_API_KEY=` 后。这个密钥是你给本服务设置的访问密码；后续填写到 New API 渠道。不要将密钥提交到 Git。

默认监听 `0.0.0.0:8080`，默认 2 个计算线程；可在 `.env` 设置 `ASR_PORT`、`ASR_BIND_HOST`、`ASR_CPU_THREADS`。CPU 识别较慢，建议先用短录音测试。4 核服务器默认留出部分计算资源给其他应用。

启动：

```bash
docker compose -f docker-compose.yml up -d --build api
docker compose -f docker-compose.yml logs --tail=100 -f api
```

显式指定 `-f` 可避免旧版 `docker-compose.override.yml` 自动覆盖。首次构建下载 CPU 版 PyTorch 及依赖，首次启动下载约 4.4 GiB 模型至 `hf_cache` 卷。模型加载完成后才接收请求。推荐至少 16 GiB 内存，并为镜像和模型预留足够磁盘；实际内存和性能以录音测试为准。

如果 Hugging Face 无法访问，可设置 `.env` 的 `HF_ENDPOINT` 为你信任的兼容模型仓库，或自行下载官方模型到缓存。`ASR_MODEL_PATH` 也支持容器内的本地模型目录；使用本地目录时需自行配置对应挂载。

检查：

```bash
curl -fsS http://127.0.0.1:8080/readyz
docker compose -f docker-compose.yml ps
```

就绪后返回 `{"status":"ready"}`。`/healthz` 是进程存活接口。启动失败时容器按重启策略重试，不会伪报就绪。

对外访问需云服务器安全组 / 端口映射放行对应端口；建议仅允许 New API 服务器访问，公网使用时通过 HTTPS 反向代理。新配置只启动本服务，不会停止原服务器上的其他应用。

## 2. 直接测试服务

下面交互读取密钥，不在命令历史中写明文。替换录音路径：

```bash
read -rsp 'ASR service key: ' ASR_TEST_KEY; echo
curl -sS http://127.0.0.1:8080/v1/audio/transcriptions \
  -H "Authorization: Bearer ${ASR_TEST_KEY}" \
  -F 'model=whisper-1' \
  -F 'language=zh' \
  -F 'file=@/data/test.mp3'
unset ASR_TEST_KEY
```

返回格式示例（不是实际识别结果）：

```json
{"text":"你好，这是一段录音。"}
```

`whisper-1` 是网关兼容别名，实际运行的仍是 Qwen3-ASR；也接受 `model=qwen3-asr-1.7b`。MP3、WAV、M4A、FLAC、OGG 等通过 FFmpeg 解码，不根据扩展名判断。只接受上传文件，不接受远程录音 URL。

## 3. 接入 New API

New API 官方支持 [POST /v1/audio/transcriptions](https://github.com/QuantumNous/new-api/blob/main/router/relay-router.go)。新增渠道，具体字段名称随版本可能不同：

| 字段 | 内容 |
| --- | --- |
| 类型 | OpenAI / OpenAI 兼容 |
| 名称 | 录音转文字 |
| 代理地址 / Base URL | `http://本服务IP:8080`，不要附加 `/v1` |
| 密钥 | 本服务 `.env` 中的 `ASR_API_KEY` |
| 模型 | 先填 `whisper-1`，无需模型映射 |
| 分组 | 业务程序令牌有权使用的分组 |

如果 New API 在另一个 Docker 容器中，`127.0.0.1` 指的是它自己。使用可达的宿主机 / 内网 IP，或配置共同 Docker 网络后通过服务名访问。还需按你的 New API 版本设置模型计费 / 倍率及令牌权限。

默认渠道聊天测试可能调用 `/v1/chat/completions`，本服务不提供该接口，应使用音频请求验证。业务程序使用 **New API 令牌**：

```bash
read -rsp 'New API token: ' NEWAPI_TEST_TOKEN; echo
curl -sS http://你的NewAPI地址/v1/audio/transcriptions \
  -H "Authorization: Bearer ${NEWAPI_TEST_TOKEN}" \
  -F 'model=whisper-1' \
  -F 'language=zh' \
  -F 'file=@/data/test.mp3'
unset NEWAPI_TEST_TOKEN
```

New API、前置代理及调用程序需要允许相应上传大小，并将超时设为可覆盖实际识别耗时的值，例如 600 秒起。长录音 CPU 推理可能需要更长超时，请实际测量。

## 4. API 约定

`GET /v1/models`、`POST /v1/audio/transcriptions` 要求 `Authorization: Bearer <ASR_API_KEY>`。健康接口无需鉴权。根路径没有管理页面。

| 参数 | 说明 |
| --- | --- |
| `file` | 必填，multipart 音频文件 |
| `model` | 必填，`whisper-1` 或 `ASR_MODEL_NAME` |
| `language` | 可选，`zh`、`en` 等；不传则自动检测 |
| `prompt` | 可选，上下文 / 术语提示，最多 2000 字符 |
| `response_format` | `json`（默认）、`text`、`verbose_json`、`srt`、`vtt` |
| `temperature` | 仅支持 `0` |
| `stream` | 仅支持 `false`；当前为同步文件转录 |
| `timestamp_granularities[]` | 仅支持 `segment`，不支持逐字时间戳 |

`verbose_json` 返回 `task / language / duration / text / segments`。时间来自 VAD 检测的语音区间，是近似分段时间，不是逐字对齐；静音录音可返回空文字。

默认最大上传 25 MiB，最长 1800 秒；超限返回 `413`，不会静默截断。格式错误 / 无法解码返回 `400`，鉴权失败返回 `401`。单进程每次处理一条录音，忙时返回 `503` 和 `Retry-After: 5`，调用方稍后重试。

上传录音仅在临时目录处理，请求结束后清理；不提供任务持久队列或录音存储。密钥和转录正文不写入应用日志。当前没有说话人分离、业务质检或语音合成。

## 5. 代码及测试

| 文件 | 作用 |
| --- | --- |
| `app/server.py` | 生产启动入口 |
| `app/main.py` | API、鉴权、上传限制、并发保护、输出格式 |
| `app/qwen_asr.py` | FFmpeg 解码、VAD 分段、本地 Qwen 推理 |
| `app/config.py` | `.env` 配置及校验 |
| `Dockerfile` / `docker-compose.yml` | Ubuntu 22.04 CPU 容器部署 |
| `requirements-linux.lock` | Linux x86_64 / Python 3.10 固定依赖版本 |
| `tests/` | HTTP 和音频解码回归测试 |

```bash
python -m pip install -c requirements-linux.lock -r requirements-dev.txt
python -m pytest tests -q
```

接口自动测试使用替身识别器；真实服务测试记录见 [VALIDATION.md](VALIDATION.md)。模型接口依据 [Qwen3-ASR 官方项目](https://github.com/QwenLM/Qwen3-ASR)。

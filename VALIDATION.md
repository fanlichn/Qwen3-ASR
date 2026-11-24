# 部署与验收记录

日期：2026-11-24。

## 部署结果

- 仓库：`https://github.com/fanlichn/Qwen3-ASR`，分支 `main`；服务器目录：`/data/Qwen3-ASR`。
- Ubuntu 22.04.5 LTS，4 个 CPU，约 30.4 GiB 内存。实机未检测到 NVIDIA GPU，按用户确认采用 **CPU 部署**。
- Docker 29.4.3、Compose v5.1.3，容器 `qwen3-asr-api-1`，监听 `0.0.0.0:8080`。
- PyTorch `2.6.0+cpu`，Qwen3-ASR-1.7B，float32，默认 2 个计算线程。
- 官方模型版本：`7278e1e70fe206f11671096ffdd38061171dd6e5`。两个权重分片均核对官方 SHA256，通过后置于独立模型缓存卷的标准 Hugging Face `hub` 目录。
- 独立随机服务密钥存于服务器 `.env`，权限 `600`，没有打印到聊天、命令输出或日志。

## 修改内容

- 当前项目仅保留语音转文字：`app/main.py` 提供上传、鉴权、模型发现、输出格式、大小限制及单任务并发控制；`app/qwen_asr.py` 完成解码、VAD 分段和模型推理。
- `app/config.py`、`.env.example`、`Dockerfile`、`docker-compose.yml` 改为 CPU 默认配置，无 NVIDIA 容器依赖；`requirements-linux.lock` 固定 CPU 版依赖。
- 原业务后台、OSS、Worker、质检及 MySQL 连接模块移出当前版本。
- 修改前文件备份在原本地工作区 `.asr-api-backup-20261124-141955/`，不会上传 Git 或进入镜像。
- 没有删除数据库、录音、旧镜像或已有 Docker 数据卷，没有停止其他应用；只重启了本次新建的语音服务。

## 实际执行的主要命令

通过指定 SSH 密钥连接后执行，密钥内容和服务令牌均未写入命令文本：

```bash
git clone https://github.com/fanlichn/Qwen3-ASR.git /data/Qwen3-ASR
cd /data/Qwen3-ASR
git pull --ff-only
docker compose -f docker-compose.yml config --quiet
docker compose -f docker-compose.yml build api
docker compose -f docker-compose.yml up -d --no-build api
curl -fsS http://127.0.0.1:8080/readyz
docker compose -f docker-compose.yml ps
docker compose -f docker-compose.yml logs --tail=25 api
docker compose -f docker-compose.yml restart api
docker stats --no-stream
```

另通过受控脚本下载和校验官方模型、生成 `.env`、测试真实 HTTP 请求，以及在临时 Ubuntu 容器中执行 `python3 -m pytest tests -q -p no:cacheprovider`。临时测试容器使用 `--rm`，未在生产容器里安装测试依赖。

## 验证结果

| 验证 | 结果 |
| --- | --- |
| Windows 本地自动测试 | 40 项通过 |
| Ubuntu 22.04 / Python 3.10 实际镜像内自动测试 | 40 项通过 |
| 镜像构建及 `pip check` | 通过，无依赖冲突 |
| 容器内 PyTorch / Qwen 导入、真实静音 VAD | 通过 |
| 模型加载及 `/readyz` | 通过，容器 healthy |
| 在线 HTTP 检查 | 13 项通过 |
| 重启后重复在线检查 | 13 项再次通过 |
| 重启后模型缓存复用 | 通过 |

13 项在线检查包括：就绪、无密钥拒绝、错误密钥拒绝、模型发现、真实中文识别、真实英文识别、空音频、损坏文件、未知模型、上传超限、并发忙碌返回 503、已接受任务完成、静音返回空文字。

真实识别使用 [Qwen 官方项目](https://github.com/QwenLM/Qwen3-ASR) 的公开音频 `asr_zh.wav`、`asr_en.wav`。第一次在线测试结果：

- 中文：`甚至出现交易几乎停滞的情况。`，请求约 **3.65 秒**。
- 英文：15.051 秒音频自动检测为 `en`，返回带分段时间的英文转录，请求约 **12.72 秒**。
- 已就绪、空闲状态下，容器内存约 **8.7 GiB**，主机剩余磁盘约 **32 GiB**。单次样本耗时不能代表所有长录音性能。

测试客户端有 1 条上游弃用提示；模型启动时有生成参数提示，均未导致识别或接口测试失败。

## 服务器上复测

验收脚本和公开测试音频已放在服务器的忽略目录 `.deployment`，不会进入 Git：

```bash
cd /data/Qwen3-ASR
python3 .deployment/smoke.py
```

该脚本自动读取本机 `.env`，不打印密钥，输出每项状态和公开样本识别结果。

内部访问地址：`http://172.16.2.187:8080`。同机使用 `http://127.0.0.1:8080`；其他机器需要具备相应内网 / VPN 路由。

## 验证边界

- 用户确认该主机没有公网 IP，SSH 通过 VPN NAT 进入；**按要求忽略公网访问，不修改 NAT 或端口映射**。
- 未接入用户的实际 New API 实例，尚不能宣称网关端到端测试通过；当前已验证兼容协议的真实 HTTP 调用。New API 配置方法见 `README.md`。
- 默认一次处理一条录音，忙时返回 `503`，不包含排队及持久任务存储。业务录音准确率还需使用实际样本验收。

# Nginx 与 Cloudflare 域名访问

域名：`asr.lakdriya.com`；源站 EIP：`47.88.86.144`。SSH 使用 `root@47.88.86.144:22`，沿用原密钥，不在仓库保存私钥。

链路：`调用程序 → HTTPS / Cloudflare → HTTP:80 / Nginx → 127.0.0.1:8080 / 语音 API`。

## Cloudflare

1. DNS 的 `asr` A 记录指向 `47.88.86.144`，开启代理（橙色云朵）。
2. 源站按本次要求不安装证书，所以此主机名需要 **Flexible** 回源模式。建议通过“规则 → 配置规则”仅对 `主机名等于 asr.lakdriya.com` 设置 SSL 为 Flexible，避免影响同区域内其他网站。若整站都采用相同回源方式，也可在 SSL/TLS 概述设置。
3. 如需 HTTP 自动跳转 HTTPS，在 Cloudflare 边缘配置；源站 Nginx 不添加 HTTPS 跳转，避免 Flexible 重定向循环。
4. API 路径不应启用缓存或要求浏览器交互的挑战。如果自定义规则覆盖 `/v1/*`，需为 API 请求设置相应例外。

这种方式由 Cloudflare 提供客户端 HTTPS，Cloudflare 到源站走 HTTP，源站不需要证书。模式说明见 [Cloudflare Flexible](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/flexible/)，按主机名单独配置见 [加密模式说明](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/)。

## Ubuntu 主机配置

配置文件：[nginx-asr.conf](nginx-asr.conf)。安装前保留现有 Nginx 配置备份；不要覆盖其他站点。

```bash
sudo apt-get update
sudo apt-get install -y nginx
cd /data/Qwen3-ASR
sudo install -m 644 deploy/nginx-asr.conf /etc/nginx/sites-available/asr.lakdriya.com.conf
sudo ln -s /etc/nginx/sites-available/asr.lakdriya.com.conf /etc/nginx/sites-enabled/asr.lakdriya.com.conf
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

若链接已经存在，检查目标是否正确，无需重复创建。以后配置修改后，先 `nginx -t` 再 reload。安全组需要允许 Cloudflare 回源访问 TCP 80。本方案不监听 443，也不申请、安装任何证书。

Nginx 上传上限 26 MiB，为接口的 25 MiB 文件和 multipart 字段预留空间；保留 Authorization 请求头和完整接口路径；不缓存转录结果，不记录访问日志；源站读取超时 3600 秒。

## 验证与 New API

```bash
# 源站，不经过 Cloudflare：
curl -fsS -H 'Host: asr.lakdriya.com' http://127.0.0.1/readyz
# 完整域名链路：
curl -fsS https://asr.lakdriya.com/readyz
```

成功返回 `{"status":"ready"}`。未携带密钥请求 `/v1/models` 返回 401，属于正常鉴权。

New API 渠道使用：

| 字段 | 内容 |
| --- | --- |
| 类型 | OpenAI 兼容 |
| API 地址 | `https://asr.lakdriya.com`，末尾不加 `/v1` |
| 模型 | `whisper-1`，实际识别仍为 Qwen3-ASR |
| 密钥 | 服务器 `/data/Qwen3-ASR/.env` 的 `ASR_API_KEY` 值 |

## 长录音超时

Cloudflare 代理有独立的响应等待上限；当前官方文档的默认 Proxy Read Timeout 为 **125 秒**。Nginx 的 3600 秒不能覆盖 Cloudflare 边缘限制，CPU 处理长录音时可能收到 524。[Cloudflare 官方限制](https://developers.cloudflare.com/fundamentals/reference/connection-limits/)

需要更长同步请求时，可让 New API 通过可信内网直接访问当前 `172.16.2.187:8080`（同机使用 `127.0.0.1:8080`），或后续单独实现异步提交 / 查询。不要把短录音的域名测试结果当作长录音均可完成的保证。

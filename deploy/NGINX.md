# Nginx 与 Cloudflare 域名访问

本文件使用占位域名 `asr.example.com`，不包含实际服务器入口或内网地址。部署时替换为你的语音服务域名。

链路：`调用程序 → HTTPS / Cloudflare → HTTP:80 / Nginx → 127.0.0.1:8080 / 语音 API`。

## Cloudflare

1. 创建指向源站公网 IP 的 A 记录，开启代理（橙色云朵）。
2. 不安装源站证书时，对该主机名使用 **Flexible** 回源模式。通过“规则 → 配置规则”，设置条件为 `(http.host eq "asr.example.com")`，SSL 为 Flexible，避免影响其他主机名。
3. 如需 HTTP 自动跳转 HTTPS，在 Cloudflare 边缘配置；源站 Nginx 不添加 HTTPS 跳转，避免重定向循环。
4. API 路径不应启用缓存或要求浏览器交互的挑战。不要为了接口测试关闭整站的安全防护。

客户端到 Cloudflare 使用 HTTPS，Cloudflare 到源站使用 HTTP，源站无需证书。[Cloudflare Flexible 说明](https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/flexible/)

## Ubuntu 主机配置

先备份已有 Nginx 配置。模板文件为 [nginx-asr.conf](nginx-asr.conf)，部署时替换占位域名：

```bash
sudo apt-get update
sudo apt-get install -y nginx
cd /data/Qwen3-ASR
ASR_DOMAIN=asr.example.com  # 改为你的实际域名
sed "s/asr\.example\.com/${ASR_DOMAIN}/g" deploy/nginx-asr.conf | sudo tee "/etc/nginx/sites-available/${ASR_DOMAIN}.conf" >/dev/null
sudo ln -s "/etc/nginx/sites-available/${ASR_DOMAIN}.conf" "/etc/nginx/sites-enabled/${ASR_DOMAIN}.conf"
sudo nginx -t
sudo systemctl enable --now nginx
sudo systemctl reload nginx
```

若链接已经存在，检查目标是否正确，不必重复创建。不要覆盖其他站点。修改配置后先 `nginx -t` 再 reload。安全组需要允许 Cloudflare 回源访问 TCP 80；本方案不监听 443，不申请或安装证书。

Nginx 上传上限 26 MiB，为接口的 25 MiB 文件和 multipart 字段预留空间；保留 Authorization 请求头及完整路径；不缓存结果，不记录访问日志；源站读取超时 3600 秒。

## 验证与 New API

```bash
# 将示例域名替换为实际域名。
curl -fsS -H 'Host: asr.example.com' http://127.0.0.1/readyz
curl -fsS https://asr.example.com/readyz
```

就绪返回 `{"status":"ready"}`。根路径 `/` 没有管理页面，JSON 404 是预期行为；不带密钥请求 `/v1/models` 返回 401，也是预期鉴权行为。

New API 渠道类型使用 OpenAI 兼容，API 地址为 `https://你的语音服务域名`，不加 `/v1`；模型为 `whisper-1`（实际运行 Qwen），密钥为服务器 `.env` 中的 `ASR_API_KEY` 值。

若默认 Python `urllib` 客户端收到 Cloudflare 403 / 1010，为客户端设置描述性的应用标识，例如 `User-Agent: qwen-asr-client/1.0`，无需伪装成浏览器或关闭整站防护。[Cloudflare 官方说明](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/cloudflare-1xxx-errors/error-1010/)

## 长录音超时

Cloudflare 代理有独立的响应等待上限，当前官方文档默认 Proxy Read Timeout 为 **125 秒**。Nginx 的 3600 秒无法覆盖边缘限制，CPU 处理长录音可能收到 524。[Cloudflare 官方限制](https://developers.cloudflare.com/fundamentals/reference/connection-limits/)

更长的同步请求可通过可信内网直接连接 API，或后续设计异步提交 / 查询；短录音的域名测试不能代替长录音性能验证。

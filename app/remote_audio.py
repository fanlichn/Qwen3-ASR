"""Bounded HTTPS downloads from operator-approved hosts, with pinned public DNS."""

import http.client
import ipaddress
from pathlib import Path
import socket
import ssl
from threading import Event, Timer
import time
from urllib.parse import quote, urlsplit

from fastapi import HTTPException

from .config import Settings


def validate_url(url: str, allowed_hosts: str):
    if not url or len(url) > 16384 or any(ord(char) <= 32 or ord(char) == 127 for char in url):
        raise HTTPException(400, 'Invalid audio URL')
    try:
        parts = urlsplit(url)
        hostname = (parts.hostname or '').encode('idna').decode('ascii').lower().rstrip('.')
        port = parts.port
    except (ValueError, UnicodeError):
        raise HTTPException(400, 'Invalid audio URL') from None
    if (parts.scheme != 'https' or not hostname or port not in (None, 443)
            or parts.username is not None or parts.password is not None or parts.fragment):
        raise HTTPException(400, 'Only HTTPS audio URLs without credentials on port 443 are supported')
    allowed = {host.strip().lower().rstrip('.') for host in allowed_hosts.split(',') if host.strip()}
    if not allowed:
        raise HTTPException(400, 'URL input is disabled; configure ASR_URL_ALLOWED_HOSTS')
    if hostname not in allowed:
        raise HTTPException(400, 'Audio URL host is not allowed')
    path = quote(parts.path or '/', safe="/%:@!$&'()*+,;=-._~")
    if parts.query:
        path += '?' + quote(parts.query, safe="=&%:@!$'()*+,;/?-._~")
    return hostname, path


def public_addresses(hostname: str):
    try:
        records = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(record[4][0] for record in records))
        if not addresses:
            raise ValueError('No addresses')
        for address in addresses:
            parsed = ipaddress.ip_address(address)
            actual = getattr(parsed, 'ipv4_mapped', None) or parsed
            if not actual.is_global or actual.is_multicast:
                raise ValueError('Non-public address')
    except (OSError, ValueError):
        raise HTTPException(400, 'Audio URL must resolve exclusively to public IP addresses') from None
    return addresses


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(self, hostname, addresses, deadline):
        super().__init__(hostname, timeout=10, context=ssl.create_default_context())
        self.addresses = addresses
        self.deadline = deadline
        self.download_socket = None

    def budget(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Download deadline exceeded')
        return min(remaining, 10)

    def connect(self):
        # DNS is checked once; connect to that exact IP while verifying the original TLS hostname.
        for address in self.addresses:
            raw = None
            try:
                raw = socket.create_connection((address, 443), timeout=self.budget())
                self.download_socket = raw
                raw.settimeout(self.budget())
                self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
                self.download_socket = self.sock
                return
            except (OSError, TimeoutError):
                if raw is not None:
                    raw.close()
        raise OSError('Remote connection failed')


def download_audio(url: str, destination: Path, settings: Settings):
    hostname, path = validate_url(url, settings.asr_url_allowed_hosts)
    deadline = time.monotonic() + settings.asr_url_timeout_seconds
    addresses = public_addresses(hostname)
    connection = PinnedHTTPSConnection(hostname, addresses, deadline)
    expired = Event()

    def abort():
        expired.set()
        stream = connection.download_socket
        if stream is not None:
            try:
                stream.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    timer = Timer(max(0, deadline - time.monotonic()), abort)
    timer.daemon = True
    response = None
    timer.start()
    limit = settings.asr_max_upload_mb * 1024 * 1024
    try:
        connection.request('GET', path, headers={
            'User-Agent': 'qwen-asr-audio-fetch/1.0', 'Accept-Encoding': 'identity',
        })
        response = connection.getresponse()
        # Do not follow redirects or forward the caller's Authorization/cookies.
        if response.status != 200:
            raise HTTPException(400, 'Remote audio must return HTTP 200; redirects are not followed')
        if response.getheader('Content-Encoding', 'identity').lower() not in ('', 'identity'):
            raise HTTPException(400, 'Compressed HTTP responses are not supported')
        length = response.getheader('Content-Length')
        if length is not None:
            if not length.isdigit():
                raise HTTPException(502, 'Invalid remote audio response')
            if int(length) > limit:
                raise HTTPException(413, 'Remote audio exceeds ASR_MAX_UPLOAD_MB')
        count = 0
        with destination.open('wb') as output:
            while True:
                connection.download_socket.settimeout(connection.budget())
                chunk = response.read1(65536)
                if not chunk:
                    break
                count += len(chunk)
                if count > limit:
                    raise HTTPException(413, 'Remote audio exceeds ASR_MAX_UPLOAD_MB')
                output.write(chunk)
        if expired.is_set():
            raise TimeoutError('Download deadline exceeded')
        if not count or (length is not None and count != int(length)):
            raise HTTPException(400, 'Remote audio is empty or incomplete')
    except HTTPException:
        raise
    except (OSError, TimeoutError, http.client.HTTPException, ValueError):
        status = 504 if expired.is_set() or time.monotonic() >= deadline else 502
        raise HTTPException(status, 'Remote audio download failed or timed out') from None
    finally:
        timer.cancel()
        if response is not None:
            response.close()
        connection.close()

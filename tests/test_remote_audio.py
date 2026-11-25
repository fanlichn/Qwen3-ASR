import io
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi import HTTPException

from app.config import Settings
from app import remote_audio


@pytest.fixture
def settings():
    return Settings(_env_file=None, asr_api_key='test-only-private-key-1234567890123456789',
                    asr_url_allowed_hosts='audio.example.com', asr_max_upload_mb=1)


@pytest.mark.parametrize('url', [
    'http://audio.example.com/a.mp3', 'file:///etc/passwd',
    'https://user:password@audio.example.com/a', 'https://audio.example.com:8443/a',
    'https://audio.example.com.evil.test/a', 'https://other.example.com/a',
    'https://audio.example.com/a#fragment', 'https://audio.example.com/a\r\nX-Foo:x',
    'https://audio.example.com:bad/a', 'https:///a',
])
def test_reject_bad_urls(url):
    with pytest.raises(HTTPException) as error:
        remote_audio.validate_url(url, 'audio.example.com')
    assert error.value.status_code == 400


def test_urls_disabled_by_default():
    with pytest.raises(HTTPException):
        remote_audio.validate_url('https://audio.example.com/a', '')


def test_signed_query_and_encoded_name_preserved():
    host, path = remote_audio.validate_url(
        'https://audio.example.com/test%281%29.mp3?Signature=abc%2Bdef%3D&Expires=123',
        ' audio.example.com ',
    )
    assert host == 'audio.example.com'
    assert path == '/test%281%29.mp3?Signature=abc%2Bdef%3D&Expires=123'


@pytest.mark.parametrize('addresses', [
    ['127.0.0.1'], ['10.0.0.1'], ['169.254.169.254'], ['100.100.100.200'],
    ['::1'], ['::ffff:127.0.0.1'], ['224.0.0.1'], ['8.8.8.8', '192.168.1.1'],
])
def test_all_dns_addresses_must_be_public(monkeypatch, addresses):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '', (ip, 443)) for ip in addresses])
    with pytest.raises(HTTPException):
        remote_audio.public_addresses('audio.example.com')


def fake_remote(monkeypatch, body=b'audio', *, status=200, length=None, encoding=None):
    headers = {}
    if length is not None:
        headers['Content-Length'] = str(length)
    if encoding is not None:
        headers['Content-Encoding'] = encoding
    response = SimpleNamespace(status=status, getheader=lambda name, default=None: headers.get(name, default),
                               read1=io.BytesIO(body).read, close=Mock())
    connection = SimpleNamespace(request=Mock(), getresponse=lambda: response, close=Mock(),
                                 download_socket=Mock(), budget=lambda: 1)
    monkeypatch.setattr(remote_audio, 'public_addresses', lambda host: ['8.8.8.8'])
    monkeypatch.setattr(remote_audio, 'PinnedHTTPSConnection', lambda *args: connection)
    return connection, response


def test_download_stream_and_cleanup(monkeypatch, tmp_path, settings):
    connection, response = fake_remote(monkeypatch, body=b'valid audio', length=11)
    target = tmp_path / 'audio'
    remote_audio.download_audio('https://audio.example.com/a?Signature=private', target, settings)
    assert target.read_bytes() == b'valid audio'
    headers = connection.request.call_args.kwargs['headers']
    assert 'Authorization' not in headers and 'Cookie' not in headers
    assert headers['Accept-Encoding'] == 'identity'
    response.close.assert_called_once()
    connection.close.assert_called_once()


@pytest.mark.parametrize('kwargs,status', [
    ({'status': 302}, 400), ({'status': 403}, 400), ({'status': 404}, 400),
    ({'length': 2 * 1024 * 1024}, 413), ({'body': b'a' * (1024 * 1024 + 1)}, 413),
    ({'body': b''}, 400), ({'body': b'a', 'length': 5}, 400),
    ({'length': 'invalid'}, 502), ({'encoding': 'gzip'}, 400),
])
def test_remote_errors_are_bounded_and_sanitized(monkeypatch, tmp_path, settings, kwargs, status):
    connection, response = fake_remote(monkeypatch, **kwargs)
    with pytest.raises(HTTPException) as error:
        remote_audio.download_audio('https://audio.example.com/private?Signature=private-secret',
                                    tmp_path / 'audio', settings)
    assert error.value.status_code == status
    assert 'private-secret' not in str(error.value.detail)
    response.close.assert_called_once()
    connection.close.assert_called_once()


def test_tls_verifies_hostname_but_connects_to_pinned_ip(monkeypatch):
    raw, tls = Mock(), Mock()
    dial = Mock(return_value=raw)
    monkeypatch.setattr(socket, 'create_connection', dial)
    import time
    connection = remote_audio.PinnedHTTPSConnection('audio.example.com', ['8.8.8.8'], time.monotonic()+10)
    context = Mock(wrap_socket=Mock(return_value=tls))
    connection._context = context
    connection.connect()
    assert dial.call_args.args[0] == ('8.8.8.8', 443)
    context.wrap_socket.assert_called_once_with(raw, server_hostname='audio.example.com')

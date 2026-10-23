"""Source-fetching boundaries, using mock HTTP and DNS only."""

import socket

import httpx
import pytest

from app import tools


@pytest.mark.parametrize("url", [
    "http://127.0.0.1/admin", "http://localhost:8000/private",
    "http://10.0.0.8/internal", "file:///etc/passwd",
    "https://name:password@example.com/", "https://example.com:8080/",
])
def test_blocks_unsafe_source_urls(url):
    with pytest.raises(ValueError):
        tools.validate_public_url(url)


def test_blocks_hostname_resolving_to_private_address(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.1", 443))])
    with pytest.raises(ValueError, match="内网"):
        tools.validate_public_url("https://source.example/article")


def mock_client(monkeypatch, handler):
    original = httpx.Client
    monkeypatch.setattr(tools.httpx, "Client", lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs))


def test_redirect_to_private_target_is_blocked_before_second_request(monkeypatch):
    visited = []

    def handler(request):
        visited.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/private"})

    mock_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="内网"):
        tools.fetch_html("https://8.8.8.8/article")
    assert len(visited) == 1


def test_fetch_html_accepts_public_html(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "text/html"}, text="<p>手机公开资料</p>"))
    assert tools.fetch_html("https://8.8.8.8/article") == "<p>手机公开资料</p>"


@pytest.mark.parametrize('encoding,header,meta', [
    ('gbk','text/html','gbk'),
    ('gb18030','text/html','gb18030'),
    ('gbk','text/html; charset=gbk',None),
    ('utf-8','text/html','utf-8'),
])
def test_html_declared_encoding_preserves_chinese_model_and_measurements(monkeypatch,encoding,header,meta):
    html = (f'<meta charset="{meta}">' if meta else '') + '<title>荣耀600 Pro评测</title><p>拍照曝光稳定，续航测试10小时。</p>'
    mock_client(monkeypatch, lambda request: httpx.Response(200,headers={'content-type':header},content=html.encode(encoding)))
    decoded = tools.fetch_html('https://8.8.8.8/article')
    assert '荣耀600 Pro评测' in decoded and '续航测试10小时' in decoded and '\ufffd' not in decoded


def test_fetch_html_rejects_non_html_response(monkeypatch):
    mock_client(monkeypatch, lambda request: httpx.Response(200, headers={"content-type": "application/json"}, json={}))
    with pytest.raises(ValueError, match="不是 HTML"):
        tools.fetch_html("https://8.8.8.8/article")


def test_large_hydrated_menu_does_not_cut_off_product_parameters(monkeypatch):
    monkeypatch.setattr(tools,'MAX_PAGE_CHARACTERS',3000)
    menu='<div class="header-comp">'+''.join(f'<a href="/phone/{i}" data-image="'+ 'x'*100+'">其他产品</a>' for i in range(30))+'</div>'
    html='<title>虚构参数页</title>'+menu+'<main><p data-hydration="'+'y'*4000+'">虚构手机后置5000万像素主摄，光学防抖。</p><a href="/spec">规格参数</a></main>'
    mock_client(monkeypatch,lambda req:httpx.Response(200,headers={'content-type':'text/html'},text=html))
    actual=tools.fetch_html('https://8.8.8.8/phone')
    assert '5000万像素' in actual and '/spec' in actual and len(actual)<3000

"""Validate public source URLs and fetch HTML for phone research."""

from __future__ import annotations

import ipaddress
import socket
import re
from urllib.parse import urljoin, urlparse

import httpx
from .execution_context import remaining_seconds, request_timeout
from bs4 import BeautifulSoup, UnicodeDammit, Comment

from .config import (
    FETCH_TIMEOUT_SECONDS,
    MAX_PAGE_BYTES,
    MAX_PAGE_CHARACTERS,
    MAX_REDIRECTS,
)


USER_AGENT = (
    "Mozilla/5.0 (compatible; PhoneResearchAgent/0.1; "
    "+https://github.com/2049004750t-cmyk/phone-research-agent)"
)


class FetchedHtml(str):
    """HTML compatible with existing parsers, with observed HTTP provenance."""
    def __new__(cls, text, *, resolved_url, redirect_chain=()):
        value = super().__new__(cls, text)
        value.resolved_url = resolved_url
        value.redirect_chain = list(redirect_chain)
        return value


class RedirectBoundary(ValueError):
    def __init__(self, reason):
        self.reason = reason
        super().__init__('来源链接重定向次数过多' if reason=='redirect_limit' else '来源链接重定向形成循环')


def _is_non_public_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value)
    return not ip.is_global


def validate_public_url(url: str) -> None:
    """Reject local/private targets before the server makes an HTTP request."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("只支持 http 或 https 来源链接")
    if parsed.username or parsed.password:
        raise ValueError("来源链接不能包含用户名或密码")
    if parsed.port not in {None, 80, 443}:
        raise ValueError("来源链接只允许使用 80 或 443 端口")

    hostname = parsed.hostname
    if not hostname:
        raise ValueError("来源链接缺少域名")
    lowered = hostname.rstrip(".").lower()
    if lowered == "localhost" or lowered.endswith((".localhost", ".local")):
        raise ValueError("不允许访问本机或内网地址")

    try:
        if _is_non_public_ip(lowered):
            raise ValueError("不允许访问本机或内网地址")
        return
    except ValueError as error:
        if "不允许" in str(error):
            raise

    try:
        addresses = socket.getaddrinfo(hostname, parsed.port or 443, type=socket.SOCK_STREAM)
    except socket.gaierror as error:
        raise ValueError("来源链接域名无法解析") from error

    for address in addresses:
        ip_value = address[4][0]
        if _is_non_public_ip(ip_value):
            raise ValueError("不允许访问解析到内网的地址")


def fetch_html(url: str, *, redirect_limit: int | None = None) -> str:
    """Stream decoded HTML, stopping over the byte limit and closing each response."""
    current_url = url
    redirect_chain = []
    seen = {url}
    limit = MAX_REDIRECTS if redirect_limit is None else min(MAX_REDIRECTS, max(0, redirect_limit))
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}

    with httpx.Client(timeout=request_timeout(FETCH_TIMEOUT_SECONDS), follow_redirects=False) as client:
        for redirect_count in range(limit + 1):
            remaining_seconds()
            client.timeout=httpx.Timeout(request_timeout(FETCH_TIMEOUT_SECONDS))
            validate_public_url(current_url)
            with client.stream("GET", current_url, headers=headers) as response:
                if response.is_redirect:
                    if redirect_count == limit:
                        raise RedirectBoundary('redirect_limit')
                    location = response.headers.get("location")
                    if not location:
                        raise ValueError("来源链接返回了无目标的重定向")
                    target = urljoin(current_url, location)
                    validate_public_url(target)
                    if target in seen:
                        raise RedirectBoundary('redirect_loop')
                    seen.add(target)
                    redirect_chain.append({'from':current_url,'to':target,'kind':'http'})
                    current_url = target
                    continue

                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                    raise ValueError("链接返回的不是 HTML 页面")
                content = bytearray()
                for chunk in response.iter_bytes():
                    remaining_seconds()
                    if len(content) + len(chunk) > MAX_PAGE_BYTES:
                        raise ValueError("来源页面过大，已停止读取")
                    content.extend(chunk)
                # httpx defaults to UTF-8 when HTTP has no charset. Publishers
                # may instead declare GBK/GB18030 in HTML; honor that declaration
                # before decoding, or Chinese model names silently become junk.
                declared = re.search(r"charset\s*=\s*[\"']?([\w.-]+)", content_type, re.I)
                decoded = UnicodeDammit(bytes(content), is_html=True,
                    known_definite_encodings=[declared[1]] if declared else [])
                text = decoded.unicode_markup
                if text is None:
                    raise ValueError("来源页面编码无法识别")
                if len(text) > MAX_PAGE_CHARACTERS:
                    # Large publishers often put >100KB of scripts/navigation
                    # before the article. Preserve readable body before applying
                    # the character cap; the streaming byte cap still applies.
                    soup = BeautifulSoup(text, "html.parser")
                    navigation = BeautifulSoup("<nav></nav>", "html.parser")
                    seen_links = set()
                    for anchor in soup.select("a[href]"):
                        href, label = anchor.get("href", ""), anchor.get_text(" ", strip=True)
                        if (not label or len(label) > 100 or len(href) > 1000 or href in seen_links
                            or href.startswith(("javascript:", "mailto:", "#"))):
                            continue
                        seen_links.add(href)
                        link = navigation.new_tag("a", href=href)
                        link.string = label
                        navigation.nav.append(link)
                        if len(seen_links) >= 200 or len(str(navigation)) > MAX_PAGE_CHARACTERS // 4:
                            break
                    for node in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "img", "picture", "video", "source"]):
                        node.decompose()
                    for node in soup.find_all(["div", "section"]):
                        if node.attrs is None:
                            continue
                        label = " ".join(node.get("class", [])) + " " + node.get("id", "")
                        if (re.search(r"(?:^|[-_\s])(?:header|nav|menu)(?:$|[-_\s])", label, re.I)
                            and len(node.select("a[href]")) > 12):
                            node.decompose()
                    for comment in soup.find_all(string=lambda value: isinstance(value, Comment)):
                        comment.extract()
                    # Hydrated product sites carry huge image/data/style attributes
                    # on otherwise short cells. Truncating that markup used to cut
                    # off the entire specification section after the site menu.
                    for node in soup.find_all(True):
                        node.attrs = {key: value for key, value in node.attrs.items()
                                      if key in {"href", "class", "id", "itemprop", "content", "charset"}}
                    # Keep observed links for model-directed navigation. Fact
                    # extraction drops nav, so a menu cannot become phone specs.
                    nav_text = str(navigation)
                    text = str(soup)[:MAX_PAGE_CHARACTERS-len(nav_text)] + nav_text
                return FetchedHtml(text[:MAX_PAGE_CHARACTERS], resolved_url=current_url, redirect_chain=redirect_chain)

    raise ValueError("来源页面读取失败")

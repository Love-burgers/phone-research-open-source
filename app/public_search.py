"""Bounded public search; independent of chat and comparison business flows."""
from ddgs import DDGS
from .tools import validate_public_url
from .execution_context import request_timeout
from urllib.parse import urlencode, urljoin
from bs4 import BeautifulSoup
from .tools import fetch_html
SearchResult = dict[str, str]


def search_sogou_web(query: str, max_results: int = 4) -> list[SearchResult]:
    """Parse ordinary search results without fetching their destination bodies."""
    page = 'https://www.sogou.com/web?'+urlencode({'query':query})
    soup = BeautifulSoup(fetch_html(page), 'html.parser')
    results, seen = [], set()
    for heading in soup.select('h3.vr-title'):
        anchor = heading.select_one('a[href]')
        if anchor is None:
            continue
        url = urljoin(page, anchor.get('href', ''))
        try:
            validate_public_url(url)
        except ValueError:
            continue
        if url in seen:
            continue
        container = heading.find_parent('div', class_='vrwrap') or heading.parent
        snippet = container.select_one('.fz-mid')
        results.append({'title':anchor.get_text(' ',strip=True)[:240], 'url':url,
                        'snippet':snippet.get_text(' ',strip=True)[:500] if snippet else '',
                        'provider_query':query,'provider':'sogou'})
        seen.add(url)
        if len(results) >= max_results:
            break
    return results

def search_web(query: str, max_results: int = 4) -> list[SearchResult]:
    """Search public web pages without depending on a user browser session."""
    client = DDGS(timeout=request_timeout(8))
    results = client.text(query, region="cn-zh", safesearch="moderate",
                          max_results=max_results, backend="google")
    normalized: list[SearchResult] = []
    for item in results:
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url:
            continue
        try:
            validate_public_url(url)
        except ValueError:
            continue
        normalized.append(
            {
                "title": " ".join(str(item.get("title") or "").split())[:240],
                "url": url,
                "snippet": " ".join(str((item.get("body") or item.get("snippet"))or "").split())[:500],
            }
        )
    return normalized

"""Bounded public lookup for phones outside the bundled corpus, with a cache."""

import base64
from contextlib import suppress
import hashlib
import json
import logging
from pathlib import Path
import re
import tempfile
import time
from urllib.parse import parse_qs, urlencode, urlsplit, urljoin, unquote

from bs4 import BeautifulSoup

from .chat_schemas import ChatSource, PriceOption
from .phone_names import checked_today, normalized_name, parse_phone_names
from .public_search import search_web, search_sogou_web
from .retrieval import build_queries, exact_model_in_text, focus_aliases, focus_terms, source_score, variant_only_title
from .tools import fetch_html, validate_public_url, FetchedHtml, RedirectBoundary
from .config import MAX_REDIRECTS
from .evidence_rules import measured_game_parts, photo_observation, endurance_observation
from .execution_context import DeadlineExceeded

from .source_rules import OFFICIAL_DOMAINS, official_phone_url
logger = logging.getLogger(__name__)


def directory_links(phone, query):
    """Curated URLs are pointers only: facts still require a successful body fetch."""
    path = Path(__file__).resolve().parent / "knowledge_samples" / "public_source_directory.jsonl"
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, ValueError):
        return []
    rows = [row for row in rows if normalized_name(row["phone_name"]) == normalized_name(phone.name)]
    review_need = bool(re.search(r"游戏|原神|星穹铁道|帧率|实测|评测|发热", query))
    rows.sort(key=lambda row: row["kind"] != ("review" if review_need else "spec_price"))
    return [{"title": row["title"], "url": row["url"], "snippet": "来源目录线索，正文尚未核对，不包含预设答案。",
             "discovery_origin": "curated_url_directory", "purpose": row["kind"]} for row in rows[:3]]


def bing_target(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.hostname and (parsed.hostname == "bing.com" or parsed.hostname.endswith(".bing.com")) and parsed.path.startswith("/ck/"):
        value = parse_qs(parsed.query).get("u", [""])[0]
        if value.startswith("a1") and len(value) < 4096:
            raw = value[2:]
            return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8")
    return url


def search_phone_web(query: str, limit: int = 4) -> list[dict]:
    """Execute Bing only; engine changes belong to the model, with feedback."""
    # Preserve the model's query, including deliberate quotes.
    html = fetch_html("https://cn.bing.com/search?" + urlencode({"q":query,"mkt":"zh-CN","qs":"n"}))
    soup = BeautifulSoup(html, "html.parser")
    results = []
    for item in soup.select("li.b_algo"):
        anchor = item.select_one("h2 a")
        if anchor is None:
            continue
        try:
            url = bing_target(anchor.get("href", ""))
            validate_public_url(url)
        except Exception:
            continue
        snippet = item.select_one(".b_caption p") or item.find("p")
        results.append({"title":anchor.get_text(" ",strip=True)[:240],"url":url,
                        "snippet":snippet.get_text(" ",strip=True)[:500] if snippet else "",
                        "provider_query":query})
        if len(results)>=limit:
            break
    return results


def article_root(soup):
    """Prefer the article itself over publisher navigation and related stories."""
    explicit = soup.select_one('article, [itemprop="articleBody"], .article-content, .article_content, .post-content, .entry-content, .context-box, #article-content')
    if explicit and len(explicit.get_text(" ", strip=True)) >= 40:
        return explicit
    heading = soup.find('h1')
    if heading:
        for parent in heading.parents:
            prose = [p for p in parent.find_all('p') if len(p.get_text(strip=True)) >= 60
                     and sum(len(a.get_text(strip=True)) for a in p.find_all('a')) < len(p.get_text(strip=True)) * .25]
            if len(prose) >= 3:
                return parent
    return soup.find('main') or soup.body or soup


def extract_phone_excerpt(phone_name: str, html: str, need: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    if variant_only_title(phone_name, title) or re.search(r"404|page not found|页面不存在|找不到页面|访问验证|access denied", title, re.I):
        return None
    for element in soup(["script", "style", "noscript", "svg", "nav", "footer", "header", "aside"]):
        element.decompose()
    main = article_root(soup)
    body = main.get_text(" ", strip=True)
    if not exact_model_in_text(phone_name, body) or re.search(r"^.{0,80}(?:404|页面不存在|page not found)", body, re.I):
        return None
    headings = " ".join(e.get_text(" ", strip=True) for e in main.select("h1"))
    identified = exact_model_in_text(phone_name, title) or exact_model_in_text(phone_name, headings)
    wanted = normalized_name(phone_name)
    mixed = any(normalized_name(m.name) != wanted for m in parse_phone_names(title + " " + body))
    # A dedicated product/specification page can contain anonymous spec cells and
    # later comparisons; a comparison page needs a model name in each excerpt.
    single_page = exact_model_in_text(phone_name, title) and all(
        normalized_name(m.name) == wanted for m in parse_phone_names(title))
    candidates, seen = [], set()
    aliases = focus_aliases(need)
    for element in main.select("h1, h2, h3, p, li, td, dt, dd, div, span"):
        if element.name in {"h1", "h2", "h3"}:
            continue  # Section headings and article-link titles aren't test results.
        if any(re.search(r"(?:^|[-_\s])(?:related|recommend|sidebar|comment|hot-list)(?:$|[-_\s])",
                         " ".join(parent.get("class", [])) + " " + parent.get("id", ""), re.I)
               for parent in [element, *list(element.parents)[:4]]):
            continue
        # Avoid combining an entire page or several unrelated specification cells.
        if element.name in {"div", "span", "li", "td"} and element.find(["div", "p", "li", "table"]):
            continue
        value = " ".join(element.get_text(" ", strip=True).split())
        linked = sum(len(a.get_text(" ", strip=True)) for a in element.select("a[href]"))
        if linked and linked >= len(value) * .7:
            continue
        # Specification values often live separately from their short field label.
        # Pair only nearby bounded containers; never borrow a label from another phone.
        if len(value) <= 96 or re.search(r"\d+\s*Hz", value, re.I):
            for parent in list(element.parents)[:5]:
                if len(parent.get_text(" ", strip=True)) > 500:
                    break
                labels = parent.select(".attr-item-name, dt, th")
                label = labels[0] if len(labels) == 1 else None
                if label and label is not element:
                    key = label.get_text(" ", strip=True)
                    if 1 <= len(key) <= 30 and key not in value:
                        value = key + "：" + value
                        break
        if not 12 <= len(value) <= 650 or value in seen:
            continue
        seen.add(value)
        names = parse_phone_names(value)
        if any(normalized_name(m.name) != wanted for m in names) or variant_only_title(phone_name, value):
            continue
        exact = exact_model_in_text(phone_name, value)
        if not single_page:
            heading = element.find_previous(["h1", "h2", "h3", "h4"])
            heading_names = parse_phone_names(heading.get_text(" ", strip=True)) if heading else []
            if heading_names and any(normalized_name(m.name) != wanted for m in heading_names):
                continue
            # A comparison such as "X的亲民版" describes a different unnamed
            # subject, even when X is the only explicit model in this paragraph.
            if re.search(re.escape(phone_name) + r".{0,8}(?:亲民版|平替|替代品)", value, re.I):
                continue
        if not exact and (not identified or mixed and not single_page):
            continue
        if not exact and not re.search(r"刷新率|屏幕|电池|充电|摄像头|像素|镜头|处理器|芯片|CPU|GPU|骁龙|天玑|mAh|机身|重量|厚度|续航|游戏|实测|样张|原神|星穹铁道|王者|售价|零售价", value, re.I):
            continue  # Anonymous promotion/navigation text isn't phone evidence.
        score = sum(any(alias in value for alias in group) for group in aliases) * 5
        score += sum(word in value for word in ("摄像", "相机", "电池", "充电", "芯片", "CPU", "GPU", "mAh", "显示", "重量", "处理器", "售价", "小时", "Hz", "GB"))
        if re.search(r"原神|星穹铁道|王者荣耀|游戏|性能|电竞", need) and re.search(r"原神|星穹铁道|王者荣耀", value):
            score += 20
            if re.search(r"\d+(?:\.\d+)?\s*(?:帧|fps|℃|摄氏度)|分钟|半小时", value, re.I):
                score += 15
        if re.search(r"高刷|刷新率|屏幕", need) and re.search(r"刷新率|\d+\s*Hz", value, re.I):
            score += 25
        if re.search(r"拍照|影像|摄影|样张|夜景", need) and re.search(r"曝光|白平衡|对焦|噪点|肤色|动态范围|暗部|高光", value):
            score += 25
        if re.search(r"配置|参数|规格", need) and re.search(r"\d.*(?:像素|光圈|光学变焦|mAh|[wW])|OIS", value, re.I):
            score += 30
        if re.search(r"续航|电池|充电", need) and re.search(r"测试|实测|播放|耗电", value) and re.search(r"\d+(?:\.\d+)?\s*(?:小时|分钟|%|％)", value):
            score += 25
        if re.search(r"价格|预算|售价", need) and re.search(r"\d{3,5}(?:\.\s*0+)?\s*元", value):
            score += 25
        if score:
            candidates.append((score, value))
    candidates.sort(key=lambda pair: -pair[0])
    picked = []
    # Keep one useful passage for each active dimension before filling the rest.
    # Otherwise several long price/configuration cells can crowd out a photo or
    # endurance observation in a mixed-use question.
    for group in aliases:
        def dimension_score(pair):
            score, value = pair
            if group[0] == "拍照":
                score += 30 * bool(re.search(r"样张|实拍|成片|拍摄的|拍摄时", value))
                score += 20 * bool(re.search(r"曝光|白平衡|对焦|噪点|肤色|色调|动态范围|暗部|高光", value))
                if re.search(r"配置|参数|规格", need):
                    score += 60 * bool(re.search(r"\d.*(?:像素|光圈|光学变焦)|OIS", value, re.I))
            elif group[0] == "续航":
                score += 30 * bool(re.search(r"\d+(?:\.\d+)?\s*(?:小时|分钟|%|％)", value)
                                         and re.search(r"测试|实际|连续|耗电|剩余", value))
            elif group[0] == "性能":
                score += 60 * bool(re.search(r"(?:CPU|处理器)(?:型号|平台)[：:]",value,re.I))
            elif group[0] == "价格":
                score += 30 * bool(re.search(r"\d{3,5}\s*元", value))
            return score
        relevant = [(score, value) for score, value in candidates if any(alias in value for alias in group)
                    or group[0]=='性能' and re.search(r'CPU(?:型号|平台)[：:]',value,re.I)]
        best = max(relevant, key=dimension_score)[1] if relevant else None
        if best and not any(best in old or old in best for old in picked):
            picked.append(best)
    for _, value in candidates:
        if any(value in old or old in value for old in picked):
            continue
        picked.append(value)
        if len(picked) >= 6:
            break
    return " / ".join(picked)[:1800] or None


def extract_price_options(phone_name, html):
    """Read version/price pairs from a dedicated phone page, not a comparison."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    wanted = normalized_name(phone_name)
    if not exact_model_in_text(phone_name, title) or any(normalized_name(m.name) != wanted for m in parse_phone_names(title)):
        return [], ""
    for node in soup(["script", "style", "nav", "header", "footer", "aside"]):
        node.decompose()
    pattern = re.compile(r"(\d{1,2}\s*GB\s*[+/]\s*(?:\d{2,4}\s*GB|[12]\s*TB))\s*[)）]?\s*[:：]?\s*(?:售价|价格)?\s*[¥￥]?\s*(\d{3,5})(?:\.\s*0{1,2})?\s*元", re.I)
    offers, quotes = {}, []
    for element in article_root(soup).select("p, li, td, dd"):
        text = " ".join(element.get_text(" ", strip=True).split())
        if len(text) > 650 or any(normalized_name(m.name) != wanted for m in parse_phone_names(text)):
            continue
        previous_end,other_variant=0,False
        for match in pattern.finditer(text):
            label=text[previous_end:match.start()]
            previous_end=match.end()
            if variant_only_title(phone_name,label):other_variant=True
            elif exact_model_in_text(phone_name,label):other_variant=False
            if other_variant:continue
            price = int(match[2])
            if not 500 <= price <= 100000:
                continue
            variant = re.sub(r"\s+", "", match[1]).upper().replace("/", "+")
            offers[(variant, price)] = PriceOption(variant=variant, price=price)
            if not any(text in old for old in quotes):
                quotes.append(text)
    # Multiple prices for the same version are ambiguous (old/current/promotion).
    ambiguous = {variant for variant, _ in offers if sum(v == variant for v, _ in offers) > 1}
    return [offer for (variant, _), offer in offers.items() if variant not in ambiguous][:12], " / ".join(quotes)[:1800]


def phone_page_links(phone, url, html, need):
    """Expose same-model navigation links; executing them remains the model's choice."""
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    page_matches = exact_model_in_text(phone.name, title) and not variant_only_title(phone.name, title)
    # Product navigation often contains the only official parameters/price
    # link. Keep nav/header here; factual excerpt extraction still removes them.
    for node in soup(["footer", "aside"]):
        node.decompose()
    current_host = urlsplit(url).hostname
    product = re.search(r"(?:index|/)(\d{6,9})(?:\.shtml|/)", urlsplit(url).path)
    aliases = {normalized_name(a) for a in [phone.name, *phone.aliases]}
    model_keys = aliases | {normalized_name(phone.name.split(" ", 1)[-1])}
    found = {}
    for anchor in soup.find_all("a", href=True):
        text = anchor.get_text(" ", strip=True)[:240]
        target = urljoin(url, anchor["href"]).split("#", 1)[0]
        if target == url.split("#", 1)[0] or not text or variant_only_title(phone.name, text):
            continue
        names = parse_phone_names(text)
        if any(normalized_name(m.name) not in aliases for m in names):
            continue
        same_name = exact_model_in_text(phone.name, text)
        if official_phone_url(phone.brand, url) and normalized_name(text) in model_keys:
            same_name = True  # Maker navigation often omits its own brand prefix.
        official_product = (same_name and official_phone_url(phone.brand, url)
                            and official_phone_url(phone.brand, target))
        contextual = (page_matches and current_host == urlsplit(target).hostname and
                      bool(re.search(r"评测|实测|参数|规格", text)) and
                      (any(normalized_name(unquote(segment)) in model_keys for segment in urlsplit(target).path.split('/') if segment)
                       or product and product[1] in urlsplit(target).path))
        if not same_name and not contextual:
            continue
        if not official_product and not re.search(r"评测|实测|参数|规格|review|param", text + " " + target, re.I):
            continue
        if source_market(target, text) == "其他地区":
            continue
        score = (20 if re.search(r"评测|实测|review|article", text + target, re.I) else 0)
        if re.search(r"原神|星穹铁道|游戏|帧率", need) and re.search(r"评测|实测|review|article", text + target, re.I):
            score += 20
        found[target] = (score, {"title": text, "url": target, "status": "body_unverified", "found_on": url})
    result = []
    for _, candidate in sorted(found.values(), key=lambda pair: -pair[0])[:6]:
        try:
            validate_public_url(candidate["url"])
        except ValueError:
            continue
        result.append(candidate)
    return result


def source_market(url: str, title: str) -> str:
    """Reject known non-mainland official variants; mark unconfirmed review regions."""
    parsed = urlsplit(url)
    host, path = (parsed.hostname or "").lower(), parsed.path.lower()
    if host in {"apple.com", "www.apple.com"} or re.search(r"/(?:tw|hk|global|uk|de|es|fr|in|us)/", path):
        return "其他地区"
    if host.endswith((".com.tw", ".hk")) or re.search(r"台湾|台灣|香港|国际版|國際版|global version", title, re.I):
        return "其他地区"
    if host == "support.apple.com" and path.startswith("/zh-cn/"):
        return "中国大陆"
    if host in {"www.apple.com.cn", "apple.com.cn", "www.mi.com", "mi.com", "m.mi.com"}:
        return "中国大陆"
    if host.endswith((".vivo.com.cn", ".iqoo.com")) or host in {"vivo.com.cn", "iqoo.com"}:
        return "中国大陆"
    if path.startswith("/cn/") and any(host == domain or host.endswith('.' + domain) for domain in
                                      ("oneplus.com", "oppo.com", "honor.com", "realme.com", "samsung.com", "consumer.huawei.com")):
        return "中国大陆"
    return "地区未确认"


def preview_kind(title: str) -> str:
    if re.search(r"评测|测评|实测|游戏测试|review", title, re.I):
        return "review"
    if re.search(r"购买|价格|售价|buy|shop", title, re.I):
        return "price"
    return "spec"


def page_kind(title: str, excerpt: str) -> str:
    # A review search result is not itself a measured result.
    # A photographic observation can be qualitative; it need not contain the
    # literal word 样张. Require multiple concrete aspects plus observed behavior,
    # rather than upgrading an advertising adjective to a measured conclusion.
    photo_result = photo_observation(excerpt) and re.search(r"拍摄|样张|实拍|照片|摄像头|主摄|镜头", excerpt)
    if preview_kind(title) == "review" and (measured_game_parts(excerpt) or photo_result or endurance_observation(excerpt)
            or re.search(r"实拍|样张|续航测试|续航实测|充电实测|充电测试|屏幕实测|亮度实测", excerpt)):
        return "review"
    if preview_kind(title) == "price" and re.search(r"价格|售价|RMB|人民币|元", excerpt):
        return "price"
    return "spec"


def published_date(html: str) -> str | None:
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one('meta[property="article:published_time"], meta[name="publishdate"], time[datetime], [itemprop="datePublished"]')
    value = (node.get("content") or node.get("datetime") or "") if node else ""
    match = re.match(r"\d{4}-\d{2}-\d{2}", value)
    return match[0] if match else None


def search_redirect_target(url, html):
    parsed=urlsplit(url)
    if parsed.hostname not in {'www.sogou.com','sogou.com'} or parsed.path!='/link':
        return None
    for meta in BeautifulSoup(html,'html.parser').select('meta[http-equiv]'):
        if meta.get('http-equiv','').casefold()!='refresh':continue
        match=re.search(r'url\s*=\s*(.+)',meta.get('content',''),re.I)
        if not match:continue
        target=urljoin(url,match[1].strip().strip("'\""))
        try:validate_public_url(target)
        except ValueError:continue
        if target!=url:return target
    return None


class DiscoveryResults(list):
    lookup_origin = "network"

    def __init__(self, values=(), **details):
        super().__init__(values)
        self.__dict__.update(details)


class PublicPhoneSources:
    def __init__(self, cache_dir: Path, searcher=None, fetcher=None):
        self.cache_dir = cache_dir
        self.searcher = searcher if searcher is not None else search_phone_web
        self.custom_searcher = searcher is not None
        self.custom_fetcher = fetcher is not None
        self.fetcher = fetcher if fetcher is not None else fetch_html

    def search_backend(self, backend):
        if backend == 'google':
            return search_web
        if self.custom_searcher:
            return self.searcher
        return search_sogou_web if backend == 'sogou' else self.searcher

    def _read_cache(self, path, *, raw=False):
        try:
            if time.time() - path.stat().st_mtime < 6 * 3600:
                text = path.read_text(encoding="utf-8")
                value = text if raw else json.loads(text)
                if isinstance(value,dict) and value.get('expires_at',float('inf')) <= time.time():
                    return None
                return value
        except (OSError, ValueError, TypeError):
            pass
        return None

    def _write_cache(self, path, value):
        temporary = None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=self.cache_dir,
                                             suffix=".tmp", delete=False) as output:
                temporary = Path(output.name)
                output.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False))
            temporary.replace(path)
        except OSError:
            pass  # Cache failures must not discard successfully retrieved data.
        finally:
            if temporary is not None:
                with suppress(OSError):
                    temporary.unlink(missing_ok=True)

    def _cached_result(self, path, phone):
        cached = self._read_cache(path)
        try:
            sources = [ChatSource.model_validate(s) for s in cached["sources"]]
            for source in sources:
                validate_public_url(source.url)
            candidates = cached["candidates"][:6]
            for candidate in candidates:
                validate_public_url(candidate["url"])
            if all(s.phone_id == phone.id for s in sources):
                return {**cached, "sources": sources, "candidates": candidates, "lookup_origin": "cache"}
        except (ValueError, TypeError, KeyError):
            pass
        return None

    def discover(self, query: str, *, budget_max=None, url=None, backend="bing", refresh=False):
        """One model-chosen search or directory fetch; no hidden query strategy."""
        results = []
        if url:
            validate_public_url(url)
            requested,seen,chain,steps=url,set(),[],[]
            status=None
            while True:
                if url in seen:
                    status='redirect_loop';break
                seen.add(url)
                page_url=url
                html,origin,error=self._page(url,refresh,MAX_REDIRECTS-len(chain))
                steps.append({'url':url,'lookup_origin':origin})
                if error:
                    status=error if error in {'redirect_loop','redirect_limit'} else 'tool_error';break
                chain.extend(getattr(html,'redirect_chain',[]))
                url=getattr(html,'resolved_url',url)
                target=search_redirect_target(url,html)
                if target is None:break
                if len(chain)>=MAX_REDIRECTS:
                    status='redirect_limit';break
                chain.append({'from':url,'to':target,'kind':'search_meta'})
                url=target
            details={'requested_url':requested,'resolved_url':url,'redirect_chain':chain,'lookup_steps':steps,
                'network_operations':sum(s['lookup_origin']=='network' for s in steps),
                'cache_operations':sum(s['lookup_origin']=='cache' for s in steps)}
            details['lookup_origin']='network' if details['network_operations'] else 'cache'
            if status:
                return DiscoveryResults(status=status,fetch_feedback={'hint':'目录读取未完成；跳转边界或网页失败，未将其当正文。'},**details)
            soup = BeautifulSoup(html, "html.parser")
            for node in soup(["script", "style", "nav", "footer", "header"]):
                node.decompose()
            for anchor in soup.select("a[href]"):
                title = anchor.get_text(" ", strip=True)[:240]
                if not parse_phone_names(title):
                    continue
                target = urljoin(url, anchor["href"])
                try:
                    validate_public_url(target)
                except ValueError:
                    continue
                if source_market(target, title) == "其他地区":
                    continue
                results.append({"title": title, "url": target, "snippet": "目录机型线索，价格、版本与用途证据尚未核对。",
                                "directory_url": url, "provider_query": query, "status": "directory_lead_only"})
                if len(results) >= 10:
                    break
            if not results:self._expire_page(page_url)
            return DiscoveryResults(results,**details)
        searcher = self.search_backend(backend)
        for hit in searcher(query, 6)[:6]:
            url = str(hit.get("url") or "")
            title = str(hit.get("title") or "")[:240]
            snippet = str(hit.get("snippet") or "")[:500]
            try:
                validate_public_url(url)
            except ValueError:
                continue
            if source_market(url, title) == "其他地区":
                continue
            results.append({"title": title, "snippet": snippet, "url": url,
                            "provider_query": hit.get("provider_query", query)})
        return DiscoveryResults(results)

    def search(self, phone, query: str, *, refresh=False, backend="bing"):
        """Execute exactly the model's query; return previews without fetching bodies."""
        key = hashlib.sha256(f"agent-search-v10\n{phone.id}\n{query}\n{backend}".encode()).hexdigest()
        cache_file = self.cache_dir / (key + ".json")
        cached = self._cached_result(cache_file, phone) if not refresh else None
        if cached is not None:
            return cached
        sources, candidates, seen = [], [], set()
        search_error = None
        try:
            searcher = self.search_backend(backend)
            # Filter for the exact model after retrieving enough results; four
            # unmatching previews must not hide a useful fifth/sixth result.
            public_hits = searcher(query, 8)[:8]
        except DeadlineExceeded:
            raise
        except Exception as error:
            public_hits = []
            search_error = type(error).__name__
            logger.warning("public_search_failed backend=%s error_type=%s", backend, search_error)
        if re.search(r"官网|官方|site:|参数|规格|价格|售价", query):
            public_hits.sort(key=lambda hit: not official_phone_url(phone.brand, hit.get("url", "")))
        # Public results have equal room; saved URL pointers only supplement them.
        for hit in [*public_hits, *directory_links(phone, query)]:
            title, snippet, url = (str(hit.get(name) or "") for name in ("title", "snippet", "url"))
            if url in seen or not exact_model_in_text(phone.name, title + " " + snippet) or variant_only_title(phone.name, title):
                continue
            mixed_preview = any(normalized_name(m.name) != normalized_name(phone.name) for m in parse_phone_names(snippet))
            try:
                validate_public_url(url)
            except ValueError:
                continue
            market = source_market(url, title)
            if market == "其他地区":
                continue
            seen.add(url)
            candidates.append({"title": title[:240], "url": url, "status": "body_unverified",
                               "discovery_origin": hit.get("discovery_origin", "public_search"),
                               "purpose": hit.get("purpose", "unknown")})
            # A matching title can be a fetch target without turning an anonymous
            # preview into cited facts about this phone.
            if mixed_preview or not exact_model_in_text(phone.name, snippet):
                continue
            sources.append(ChatSource(id=phone.id + ":web-" + hashlib.sha256(url.encode()).hexdigest()[:20],
                phone_id=phone.id, title=title[:240], url=url, quote=snippet[:1800],
                checked_on=checked_today(), source_type="snippet", market=market, evidence_kind="unknown"))
        candidates = candidates[:6]
        if not candidates:
            # Some engines ignore the model suffix and return only its maker.
            # Expose actual official-directory URLs as navigation, never facts;
            # the model may open one and choose a matching product link.
            for hit in public_hits:
                url = str(hit.get("url") or "")
                if not official_phone_url(phone.brand, url) or source_market(url, str(hit.get("title") or "")) == "其他地区":
                    continue
                try:
                    validate_public_url(url)
                except ValueError:
                    continue
                if url not in {c["url"] for c in candidates}:
                    candidates.append({"title": str(hit.get("title") or "")[:240], "url": url,
                        "status": "navigation_only", "purpose": "official_directory", "discovery_origin": "public_search"})
                if len(candidates) == 2:
                    break
        feedback = {"returned_results": len(public_hits),
                    "matching_pages": sum(c.get("status") == "body_unverified" for c in candidates),
                    "public_matching_pages":sum(c.get('status')=='body_unverified' and c.get('discovery_origin')=='public_search' for c in candidates),
                    "directory_pointer_pages":sum(c.get('discovery_origin')=='curated_url_directory' for c in candidates),
                    "navigation_pages": sum(c.get("status") == "navigation_only" for c in candidates),
                    "error_type": search_error}
        if not feedback['public_matching_pages']:
            feedback["unmatched_titles"] = [h.get("title", "")[:100] for h in public_hits[:3]]
            feedback["hint"] = "本次搜索引擎未返回匹配完整型号的公开页面，可能仍有补充目录线索；目录链接不能证明游戏表现。可自主换引擎、用型号别名或限定评测来源查询，不要把本次未找到说成全网没有实测。"
        # Empty/failed searches are useful feedback too. Cache them so repeated
        # turns do not pay the same network cost; refresh remains explicit.
        self._write_cache(cache_file, {"sources": [s.model_dump() for s in sources],
            "candidates": candidates, "search_feedback": feedback,
            **({'expires_at':time.time()+300} if not feedback['public_matching_pages'] or search_error else {})})
        return {"sources": sources, "candidates": candidates, "lookup_origin": "network", "search_feedback": feedback}

    def fetch(self, phone, url: str, need: str, *, refresh=False):
        return self.fetch_result(phone, url, need, refresh=refresh)["sources"]

    def _page_path(self,url):
        key=hashlib.sha256(("page-html-v3\n"+url).encode()).hexdigest()
        return self.cache_dir/(key+'.json')

    def _expire_page(self,url):
        """An empty extraction must not reuse a six-hour raw error/captcha page."""
        path=self._page_path(url)
        page=self._read_cache(path)
        if isinstance(page,dict):
            page['expires_at']=min(page.get('expires_at',float('inf')),time.time()+300)
            self._write_cache(path,page)

    def _page(self, url, refresh, redirect_budget=None):
        """Reuse one body or short-lived failure across different question focuses."""
        raw_cache=self._page_path(url)
        failure_cache=raw_cache.with_name(raw_cache.stem+'-failure.json')
        if not refresh:
            page=self._read_cache(raw_cache)
            if isinstance(page,dict) and isinstance(page.get('html'),str):
                if redirect_budget is not None and len(page.get('redirect_chain',[]))>redirect_budget:
                    return None,'cache','redirect_limit'
                resolved=page.get('resolved_url',url)
                validate_public_url(resolved)
                return FetchedHtml(page['html'],resolved_url=resolved,redirect_chain=page.get('redirect_chain',[])),'cache',None
            failure=self._read_cache(failure_cache)
            if isinstance(failure,dict) and isinstance(failure.get('error_type'),str):
                return None,'cache',failure['error_type']
        try:
            html=(self.fetcher(url) if self.custom_fetcher or redirect_budget is None else self.fetcher(url,redirect_limit=redirect_budget))
        except RedirectBoundary as error:
            return None,'network',error.reason
        except DeadlineExceeded:
            raise  # A turn deadline is not a persistent failure of this page.
        except Exception as error:
            kind=type(error).__name__
            logger.warning('public_fetch_failed error_type=%s',kind)
            self._write_cache(failure_cache,{'error_type':kind,'expires_at':time.time()+300})
            return None,'network',kind
        self._write_cache(raw_cache,{'html':str(html),'resolved_url':getattr(html,'resolved_url',url),
            'redirect_chain':getattr(html,'redirect_chain',[])})
        with suppress(OSError):failure_cache.unlink(missing_ok=True)
        return html,'network',None

    def fetch_result(self, phone, url: str, need: str, *, refresh=False, redirect_budget=None):
        """Fetch one caller-approved URL; the agent adapter enforces URL provenance."""
        key = hashlib.sha256(f"agent-fetch-v13\n{phone.id}\n{url}\n{need}".encode()).hexdigest()
        validate_public_url(url)
        cache_file = self.cache_dir / (key + ".json")
        cached = self._cached_result(cache_file, phone) if not refresh else None
        if cached is not None:
            if redirect_budget is not None and len(cached.get('redirect_chain',[]))>redirect_budget:
                return {'sources':[],'candidates':[],'lookup_origin':'cache','status':'redirect_limit',
                    'fetch_feedback':{'hint':'缓存中的跳转路径超过剩余跳转额度，未继续读取。'}}
            return cached
        # Re-extract another focus from the same observed HTML instead of
        # fetching that URL again merely because the wording changed.
        html,origin,error_type=self._page(url,refresh,redirect_budget)
        if error_type:
            return {'sources':[],'candidates':[],'lookup_origin':origin,
                    'status':error_type if error_type in {'redirect_limit','redirect_loop'} else 'tool_error',
                    'fetch_feedback':{'error_type':error_type,
                        'hint':'正文暂不可用；复用已有证据或选择其他已发现来源，不重复请求同一失败页面。'}}
        requested_url = url
        url = getattr(html,'resolved_url',url)
        validate_public_url(url)
        provenance = {'requested_url':requested_url,'resolved_url':url,
                      'redirect_chain':getattr(html,'redirect_chain',[])}
        # Sogou exposes opaque link wrappers. Read only the literal meta
        # redirect; do not execute JavaScript or fetch an unchosen destination.
        parsed = urlsplit(url)
        if parsed.hostname in {'www.sogou.com', 'sogou.com'} and parsed.path == '/link':
            target = search_redirect_target(url,html)
            result = {'sources':[], 'candidates':[{'url':target, 'title':phone.name+' · 搜索结果目标正文',
                       'status':'body_unverified','purpose':'search_target'}] if target else [],
                      'lookup_origin':origin,'status':'redirect_only' if target else 'empty',**provenance,
                      'fetch_feedback':{'hint':'这是搜索引擎跳转页，不是评测正文；可选择打开返回的目标URL核对内容。' if target else
                                        '搜索跳转页未提供可验证目标；未执行脚本，请选择其他来源。'}}
            if not target:
                result['expires_at']=time.time()+300
                self._expire_page(requested_url)
            self._write_cache(cache_file, result)
            return result
        excerpt = extract_phone_excerpt(phone.name, html, need)
        if re.search(r"/\d+/\d+/article\.shtml(?:\?|$)", url):
            excerpt = None  # Product review index is navigation, not an original test.
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.get_text(" ", strip=True)[:240] if soup.title else phone.name
        market = source_market(url, title)
        if market == "其他地区":
            return {"sources": [], "candidates": [], "lookup_origin": origin}
        offers, price_quote = [], ""
        # A quoted retail/version price need not be an official launch price.
        # Require explicit mainland/currency context for a non-official page.
        if market == "地区未确认" and re.search(r"中国大陆|国行|人民币|\bRMB\b|国补", article_root(soup).get_text(" ", strip=True), re.I):
            market = "中国大陆"
        if market == "中国大陆":
            offers, price_quote = extract_price_options(phone.name, html)
        source_id = phone.id + ":web-" + hashlib.sha256(url.encode()).hexdigest()[:20]
        sources = [ChatSource(id=source_id,
            phone_id=phone.id, title=title, url=url, quote=excerpt[:1800], checked_on=checked_today(),
            source_type="page", market=market, published_on=published_date(html),
            evidence_kind=page_kind(title, excerpt))] if excerpt else []
        if offers:
            first = min(offers, key=lambda offer: offer.price)
            sources.append(ChatSource(id=source_id + ":prices", phone_id=phone.id, title=title + " · 版本价格",
                url=url, quote=price_quote, checked_on=checked_today(), source_type="page", market=market,
                published_on=published_date(html), evidence_kind="price", price_options=offers,
                reference_price=first.price, price_variant=first.variant))
        candidates = phone_page_links(phone, url, html, need)
        if not sources:self._expire_page(requested_url)
        self._write_cache(cache_file, {"sources": [s.model_dump() for s in sources], "candidates": candidates,**provenance,
            **({'expires_at':time.time()+300} if not sources else {})})
        return {"sources": sources, "candidates": candidates, "lookup_origin": origin,**provenance}

    def fetch_following_result(self, phone, url: str, need: str, *, refresh=False):
        """Read a chosen page through bounded, literal search redirects only.

        A redirect is transport, not a new search or a navigation suggestion.
        No scripts, unrelated links, or inferred targets are executed.
        """
        requested, seen, steps, chain = url, set(), [], []
        while True:
            if url in seen:
                result={**result,'sources':[],'status':'redirect_loop',
                    'fetch_feedback':{'hint':'搜索跳转形成循环，未继续读取；可选择其他来源。'}}
                break
            seen.add(url)
            result=self.fetch_result(phone,url,need,refresh=refresh,redirect_budget=MAX_REDIRECTS-len(chain))
            steps.append({'url':url,'resolved_url':result.get('resolved_url',url),
                'status':result.get('status') or ('ok' if result['sources'] else 'empty'),
                'lookup_origin':result.get('lookup_origin','unknown')})
            chain.extend(result.get('redirect_chain',[]))
            target = (result.get('candidates') or [{}])[0].get('url') if result.get('status')=='redirect_only' else None
            if target is None:
                break
            if len(chain)>=MAX_REDIRECTS:
                result={**result,'status':'redirect_limit','fetch_feedback':{'hint':'搜索跳转次数达到边界，未读取目标正文；可选择其他已发现来源。'}}
                break
            validate_public_url(target)
            chain.append({'from':result.get('resolved_url',url),'to':target,'kind':'search_meta'})
            url=target
        network=sum(step['lookup_origin']=='network' for step in steps)
        return {**result,'requested_url':requested,'redirect_chain':chain,'lookup_steps':steps,
            'lookup_origin':'network' if network else 'cache' if all(s['lookup_origin']=='cache' for s in steps) else result.get('lookup_origin','unknown'),
            'network_operations':network,'cache_operations':sum(s['lookup_origin']=='cache' for s in steps)}

    def collect(self, phone, need: str, *, refresh: bool = False) -> list[ChatSource]:
        topics = " ".join(focus_terms(need)) or "拍照 续航 价格"
        gaming = bool(re.search(r"游戏|原神|王者|帧率|性能|手游|电竞", need))
        if gaming:
            topics += " 游戏 帧率 发热"
        key = hashlib.sha256(f"v3\n{phone.name}\n{topics}".encode()).hexdigest()
        path = self.cache_dir / (key + ".json")
        cached = self._read_cache(path) if not refresh else None
        if cached is not None:
            try:
                cached = [ChatSource.model_validate(item) for item in cached]
                if cached and all(item.phone_id == phone.id for item in cached):
                    return cached
            except (ValueError, TypeError):
                pass
        found = {}
        queries = list(build_queries(phone.name, topics))
        if phone.brand in OFFICIAL_DOMAINS:
            queries[0] = f'{phone.name} site:{OFFICIAL_DOMAINS[phone.brand]}'
        if gaming:
            queries[1] = f'"{phone.name}" 游戏 实测 帧率 发热'
            queries.append(f'"{phone.name}" 中国大陆 购买 价格')
        for query in queries:
            try:
                hits = self.searcher(query, 4)
            except Exception as error:
                logger.warning("fallback_search_failed error_type=%s", type(error).__name__)
                continue
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                title, snippet, url = (str(hit.get(name) or "") for name in ("title", "snippet", "url"))
                if (not exact_model_in_text(phone.name, title + " " + snippet)
                    or variant_only_title(phone.name, title)):
                    continue
                try:
                    validate_public_url(url)
                    if source_market(url, title) == "其他地区":
                        continue
                except Exception:
                    continue
                found.setdefault(url, {"title": title, "snippet": snippet, "url": url})
        ranked = sorted(found.values(), key=lambda item: -(
            source_score(phone.name, topics, item["title"], item.get("snippet", ""))
            + (8 if urlsplit(item["url"]).hostname in {"www.apple.com.cn", "support.apple.com", "www.mi.com", "m.mi.com", "mi.com"} else 0)))
        if gaming:
            # Reserve room and a body fetch for measured reviews and price pages.
            # Official specification pages must not consume the whole lookup.
            reserved = [next((item for item in ranked if preview_kind(item["title"]) == kind), None)
                        for kind in ("review", "price")]
            ranked = list({item["url"]: item for item in [*filter(None, reserved), *ranked]}.values())
        sources = []
        for index, hit in enumerate(ranked[:6 if gaming else 4]):
            excerpt, published_on = None, None
            if index < (3 if gaming else 2):
                try:
                    html = self.fetcher(hit["url"])
                    excerpt = extract_phone_excerpt(phone.name, html, topics)
                    if excerpt:
                        published_on = published_date(html)
                except Exception as error:
                    logger.warning("fallback_fetch_failed error_type=%s", type(error).__name__)
            snippet = hit.get("snippet", "")
            # An exact title alone does not make a neighbouring variant's preview evidence.
            mixed_snippet = any(normalized_name(m.name) != normalized_name(phone.name)
                                for m in parse_phone_names(snippet))
            quote = excerpt or (snippet if exact_model_in_text(phone.name, snippet)
                                and not variant_only_title(phone.name, snippet) and not mixed_snippet else None)
            if not quote:
                continue
            source_id = phone.id + ":web-" + hashlib.sha256((hit["url"] + quote).encode()).hexdigest()[:20]
            sources.append(ChatSource(id=source_id, phone_id=phone.id, title=hit["title"],
                                      url=hit["url"], quote=quote[:1800], checked_on=checked_today(),
                                      source_type="page" if excerpt else "snippet",
                                      market=source_market(hit["url"], hit["title"]), published_on=published_on,
                                      evidence_kind=page_kind(hit["title"], excerpt) if excerpt else "unknown"))
        if sources:
            self._write_cache(path, [source.model_dump() for source in sources])
        return sources

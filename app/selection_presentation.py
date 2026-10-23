"""Source-backed configuration summaries for a provisional conversation shortlist."""
import re

from .source_rules import official_phone_url
from .selection_prices import storage_gb, minimum_storage, source_price_options

from .topics import SPEC_PATTERNS
from .evidence_rules import unsupported_parameters


def presentation_copy(text, hide_test_gaps=True):
    """Move generic missing-test notices out of the prose; preserve real facts."""
    if not hide_test_gaps:
        return text, []
    gap = re.compile('|'.join([
        r'(?:实测|测试)(?:资料|记录|数据|结果)?[^，。；,;]{0,8}(?:待补|待查|待核|不足|缺失|未核|没有)',
        r'(?:待补|待查|待核|暂无|尚无|缺少|还缺|没有|未查到|未找到|尚未)[^，。；,;]{0,16}(?:实测|测试)',
        r'(?:游戏|续航|拍照)表现[^，。；,;]{0,8}(?:待核|待补|待查)',
        r'(?:待|有待|需|需要|尚待|仍需)[^，。；,;]{0,6}(?:实测|测试)(?:核对|验证)?',
        r'(?:实测|测试)[^，。；,;]{0,16}(?:待|仍需|尚需|需要|需)[^，。；,;]{0,4}(?:核实|核对|验证|确认)',
        r'(?:参数|配置|标称数据)[^，。；,;]{0,12}(?:不代表|不等于|不能证明)[^，。；,;]{0,12}(?:实际|体验|表现)',
        r'不能据此下体验结论',
    ]))
    facts = re.compile(r"\d+(?:\.\d+)?\s*(?:mAh|W|Hz|GB|TB|万|亿|元|帧|fps|小时|分钟)|骁龙\s*\d|天玑\s*\d|\bA\d{2}",re.I)
    kept, notices = [], []
    for part in re.split(r"(?<=[，。；,;])",text):
        if not part:continue
        missing_test=bool(re.search(r'(?:缺|未取得|尚无)[^，。；,;]{0,16}(?:实测|测试)|(?:实测|测试)(?:依据|数据|资料)[^，。；,;]{0,8}(?:不足|缺失|未知)',part))
        if gap.search(part) or missing_test:
            split=re.split(r"(?=但是|但|不过|只是|然而)",part,maxsplit=1)
            if len(split)==2 and facts.search(split[0]) and gap.search(split[1]) and not facts.search(split[1]):
                kept.append(split[0]);notices.append(split[1].strip());continue
            if not facts.search(part):
                notices.append(part.strip());continue
        kept.append(part)
    result=''.join(kept).strip()
    if not result:
        return text, []  # A warning-only direct answer must not turn blank.
    if result[-1:] in '，,；;':result=result.rstrip('，,；;')+'。'
    return result,notices


def configuration_facts(phone, topics, sources, limit=3):
    """Return literal parameter spans, never turn them into measured outcomes."""
    body = [s for s in sources if s.phone_id == phone.id and s.source_type != "snippet"]
    groups = []
    for topic in topics:
        found = []
        ordered = sorted(body, key=lambda s: not official_phone_url(phone.brand, s.url))
        for source in ordered:
            official = official_phone_url(phone.brand, source.url)
            if found and found[0][2] and not official:
                break
            for pattern in SPEC_PATTERNS.get(topic, []):
                for match in re.finditer(pattern, source.quote, re.I):
                    value = match[0].strip()
                    if unsupported_parameters(value,[source]):
                        continue
                    if topic == '游戏' and re.fullmatch(r'A\d{2}(?:\s*Pro)?',value,re.I):
                        context=source.quote[max(0,match.start()-24):match.end()+24]
                        if not re.search(r'芯片|仿生|处理器|CPU|SoC|搭载',context,re.I):
                            continue
                    if topic == '屏幕' and re.search(r'Hz',value,re.I):
                        before=re.split(r'[。；，,/|]',source.quote[:match.start()])[-1]
                        after=re.split(r'[。；，,/|]',source.quote[match.end():])[0]
                        context=before[-30:]+value+after[:12]
                        if re.search(r'触控|采样',context) or not re.search(r'屏幕|刷新率|显示',context):
                            continue
                    if re.match(r"(?:流媒体视频播放|视频播放|音频播放)", value):
                        if not official or source.evidence_kind != "spec":
                            continue
                        value = "官方标称" + value
                    lens = next((kind for kind in ("超广角","长焦","主摄","前置") if kind in value), None)
                    quantity = re.search(r"\d+(?:\.\d+)?\s*(?:亿|万)",value)
                    key = (re.sub(r"\s|像素", "", quantity[0]),lens or "摄像头") if quantity else re.sub(r"\s", "", value).lower()
                    if any(key == old[3] or value == old[0] or value in old[0] for old in found):
                        continue
                    found.append((value, source.id, official,key))
        if topic == "拍照":
            roles={"主摄":0,"长焦":1,"超广角":2,"摄像头":3,"前置":4}
            found.sort(key=lambda item: roles.get(item[3][1],3) if isinstance(item[3],tuple) else 3)
        if topic == '续航':
            capacities=[item for item in found if re.search(r'\d\s*mAh',item[0],re.I)]
            def capacity_order(item):
                source=next(s for s in body if s.id==item[1])
                start=source.quote.find(item[0])
                before=source.quote[max(0,start-16):start]
                return (bool(re.search(r'额定|单电芯',before)),not bool(re.search(r'等效|电池容量|电池[：:]',before)))
            capacities.sort(key=capacity_order)
            other=[item for item in found if item not in capacities]
            found=[*capacities[:1],*other,*capacities[1:]]
        groups.append(found)
    # Give every requested dimension a slot before taking a second camera
    # parameter. Equivalent parameter word order must not consume two slots.
    facts, seen = [], set()
    for position in range(limit):
        for group in groups:
            if position < len(group):
                value, sid, _, key = group[position]
                if key not in seen:
                    facts.append((value,sid));seen.add(key)
                    if len(facts) >= limit:
                        return facts
    return facts


def quoted_price(phone, profile, sources):
    """Keep official and other page prices distinct, including the exact version."""
    offers = []
    for source in sources:
        if source.phone_id != phone.id:
            continue
        for option in source_price_options(source):
            if (storage_gb(option.variant) or 0) < minimum_storage(profile):
                continue
            offers.append((not (official_phone_url(phone.brand, source.url) and source.market == "中国大陆"), option.price, option.variant, source))
    if not offers:
        return None
    official = [o for o in offers if not o[0] and o[3].market == "中国大陆"]
    pool = official or offers
    conflicting = {o[2] for o in pool if len({x[1] for x in pool if x[2] == o[2]}) > 1}
    pool = [o for o in pool if o[2] not in conflicting]
    if not pool:
        return None
    within = [o for o in pool if not profile.budget_max or o[1] <= profile.budget_max]
    pool = within or pool
    chosen = next((o for o in pool if o[2] == phone.variant and o[1] == phone.price), min(pool, key=lambda o: o[1]))
    other, price, variant, source = chosen
    label = ("页面参考价" if other else "官方建议零售价" if re.search('建议零售价',source.title+source.quote) else "官网标价")
    text = f"{label}{variant} {price}元"
    if profile.budget_max and price > profile.budget_max:
        text += f"（超预算{price-profile.budget_max}元）"
    return text, source.id


def configuration_block(phone, profile, sources):
    topics = list(profile.priorities) or ["拍照", "续航", "屏幕"]
    facts = configuration_facts(phone, topics, sources)
    price = quoted_price(phone, profile, sources)
    if not facts and not price:
        return None
    parts, ids = [], []
    if price:
        parts.append(price[0]); ids.append(price[1])
    if facts:
        parts.append("、".join(value for value, _ in facts))
        ids.extend(sid for _, sid in facts)
    return phone.name + "：" + "；".join(parts) + "。", list(dict.fromkeys(ids))[:8]


def conversation_guidance(profile):
    known_photo=any(any(word in requirement for word in ('人像','夜景','远景','舞台','拍摄','拍照')) for requirement in profile.requirements)
    if "拍照" in profile.priorities and not known_photo:
        detail = "更常拍人像、夜景还是远景"
        extra = "更看重续航" if "续航" not in profile.priorities else "希望轻一点"
    elif "游戏" in profile.priorities or profile.games:
        duration=any(re.search(r'分钟|小时',r) for r in profile.requirements)
        detail = "主要玩什么游戏" if not profile.games and duration else "主要玩什么游戏、一次玩多久" if not profile.games else "一次通常玩多久" if not duration else "更在意温度还是握持手感"
        extra = "也想兼顾拍照" if "拍照" not in profile.priorities else "希望轻一点"
    elif "续航" in profile.priorities and not known_photo:
        detail = "主要刷视频、导航还是玩游戏"
        extra = "也想拍照好一点"
    else:
        return '可以继续问其中一款的具体取舍，或说“换一批”“只留第一款”。'
    return f"你{detail}？也可以补充“{extra}”，或说“换一批”“只留第一款”。"

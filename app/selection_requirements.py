"""Purchase constraints and evidence gaps, shared by cards and the tool-loop guard."""

import re
from .topics import EVIDENCE_PATTERNS

from .evidence_rules import photo_observation, endurance_observation, game_in_text, canonical_games, measured_game_parts, parameter_values
from .selection_prices import minimum_storage, storage_gb
from .selection_presentation import configuration_facts


def literal_requirements(profile, text):
    """Small unambiguous signals also work offline; no phone facts are inferred."""
    before = profile.model_dump()
    profile.games = canonical_games(profile.games)
    if re.search(r"预算|不超过|最多花|最高花|控制在|以内|取消预算|预算不限", text):
        profile.price_preference = "within_budget"
    if re.search(r"(?:想要|要买|想买|买|要).{0,8}\d{3,5}\s*(?:元|块|左右|价位).{0,5}(?:的)?手机|\d{3,5}\s*(?:元|块)?左右", text):
        profile.price_preference = "near_budget"
    count = re.search(r"推荐\s*([一二三四五1-5])\s*(?:款|台|部)", text)
    if count:
        profile.recommendation_count = int(count[1]) if count[1].isdigit() else "一二三四五".index(count[1]) + 1
    if re.search(r"不玩游戏|取消游戏|游戏不.*(?:需要|考虑)|不用.*游戏", text):
        profile.games = []
    for clause in re.split(r"[，。；;\n]|(?:但是|但|不过)", text):
        question = bool(re.search(r"吗|多少|[?？]", clause))
        if "高刷" in clause and not question:
            profile.high_refresh_required = not bool(re.search(r"不需要|不要|不用|取消|不要求|不强求", clause))
        games = [name for name in ("原神", "崩坏：星穹铁道", "王者荣耀", "和平精英", "绝区零")
                 if game_in_text(name, clause)]
        if re.search(r"不玩|不要玩|取消", clause):
            profile.games = [g for g in profile.games if g not in games]
        elif not question:
            profile.games = canonical_games([*profile.games, *games])
    return before != profile.model_dump()


def discovery_needed(profile, catalog):
    if profile.explicit_candidate_ids or profile.focus_phone_id or not profile.budget_max:
        return False
    candidates = [catalog[pid] for pid in profile.candidate_ids if pid in catalog]
    if profile.price_preference == "save_money" and candidates:
        return False
    # A ceiling isn't a command to spend it all. This detects missing price-tier
    # coverage, rather than rejecting useful cheaper alternatives as invalid.
    return not any((p.price or 0) >= profile.budget_max * .8 and
                   (not profile.priorities or profile.priorities[0] in p.topics) for p in candidates)


def local_only(text):
    return bool(re.search(r"只用本地|只用已有|不要联网|不联网|只查本地", text))


def selection_threshold(requirement):
    """Recognize explicit gates, independent of topic/model/action classification.

    A usage duration or subject alone isn't a gate. This only decides whether
    selection must be checked; it does not claim any phone meets the requirement.
    """
    return bool(re.search(r"必须|务必|只接受|硬性|不能|不可|不得", requirement)
                or re.search(r"至少|至多|最多|不超过|不低于|不高于|起码|[<>]=?|[≥≤]", requirement)
                and re.search(r"\d|[一二两三四五六七八九十百千]", requirement))


def numeric_requirement(requirement, sources):
    """Conservative physical-spec comparison; arbitrary prose stays unverified."""
    dimensions = [(r"重量|轻于|重于|克|\bg\b", 'g', None),
                  (r"厚度|毫米|mm", 'mm', 'thickness'),
                  (r"刷新率|Hz|赫兹", 'hz', 'display_refresh'),
                  (r"电池|mAh",'mah',None)]
    for wanted, unit, role in dimensions:
        if not re.search(wanted, requirement, re.I):
            continue
        limits = [v for v,u,_,_ in parameter_values(requirement) if u==unit]
        lower = bool(re.search(r"至少|不低于|大于等于|以上|>=|≥", requirement))
        upper = bool(re.search(r"不超过|至多|最多|不高于|以下|<=|≤", requirement))
        if len(limits)!=1 or lower == upper:
            return None
        values = []
        for source in sources:
            if source.source_type=='snippet':
                continue
            values.extend((v,source.url) for v,u,r,_ in parameter_values(source.quote)
                          if u==unit and (role is None or r==role))
        if len({v for v, _ in values}) != 1:
            return None
        value = values[0][0]
        ok = value >= limits[0] if lower else value <= limits[0]
        return ("配置符合" if ok else "配置不符合"), [url for _, url in values]
    return None


def topic_observations(phone_id, topic, sources, games=()):
    """Share topic-specific body observations between cards and answer fallback."""
    observations=[]
    for source in sources:
        if source.phone_id!=phone_id or source.source_type=='snippet':continue
        if topic=='游戏':
            parts=measured_game_parts(source.quote) if source.evidence_kind=='review' else []
            observations.extend((source,part) for part in parts if not games or any(game_in_text(g,part) for g in games))
            continue
        if topic not in EVIDENCE_PATTERNS:continue
        related,measured=EVIDENCE_PATTERNS[topic]
        for part in re.split(r' / |[。；]',source.quote):
            observed=photo_observation(part) if topic=='拍照' else endurance_observation(part) if topic=='续航' else bool(re.search(measured,part,re.I))
            if (re.search(related,part,re.I) and observed
                and not re.search(r'(?:没有|缺少|未做|不含|暂无)[^。；]{0,8}(?:实测|测试|样张|评测)',part)
                and (topic in {'轻便','屏幕'} or source.evidence_kind=='review')):
                observations.append((source,part.strip()))
    return observations


def recommendation_gaps(phone, profile, sources):
    """Unknown hard gates stay queryable leads, never qualified recommendations."""
    body=[s for s in sources if s.phone_id==phone.id and s.source_type!='snippet']
    gaps=[]
    if profile.budget_max and (phone.price is None or phone.price>profile.budget_max):gaps.append('预算内版本价未确认')
    if minimum_storage(profile) and (storage_gb(phone.variant) or 0)<minimum_storage(profile):gaps.append('存储版本未符合')
    labels,_=card_matches(phone.id,profile,body)
    if any('冲突' in label for label in labels):gaps.append('来源冲突未解决')
    if profile.high_refresh_required and not any(label.startswith('高刷：来源标注') and '不符合' not in label for label in labels):
        gaps.append('高刷硬要求未确认')
    for requirement in profile.requirements:
        if not selection_threshold(requirement) or minimum_storage(profile) and storage_gb(requirement):continue
        checked=numeric_requirement(requirement,body)
        if not checked or checked[0]!='配置符合':gaps.append('硬要求未确认：'+requirement)
    for topic in profile.priorities:
        if not configuration_facts(phone,[topic],body) and not topic_observations(phone.id,topic,body,profile.games):
            gaps.append(topic+'缺少相关依据')
    return gaps


def card_matches(phone_id, profile, sources):
    sources = [s for s in sources if s.phone_id == phone_id and s.source_type != "snippet"]
    labels, urls = [], []
    if "游戏" in profile.priorities or profile.games:
        reviews = [s for s in sources if s.evidence_kind == "review" and any(
                   (any(game_in_text(game, part) for game in profile.games) if profile.games else
                    re.search(r"游戏|原神|星穹铁道|王者", part)) for part in measured_game_parts(s.quote))]
        labels.append("游戏表现：有实测资料，见来源条件" if reviews else "游戏表现：待补实测")
        urls.extend(s.url for s in reviews)
        for game in profile.games:
            matched = [s for s in reviews if any(game_in_text(game, part) for part in measured_game_parts(s.quote))]
            labels.append(f"{game}：" + ("有测试记录" if matched else "待查"))
    if profile.high_refresh_required or "屏幕" in profile.priorities:
        hits = [(s,value) for s in sources for value,unit,role,_ in parameter_values(s.quote)
                if unit=='hz' and role=='display_refresh']
        rates = {max(value for source,value in hits if source.id==s.id) for s,_ in hits}
        if len(rates) > 1:
            labels.append("高刷：来源冲突（" + " / ".join(f"{n}Hz" for n in sorted(rates)) + "），待核对版本")
        else:
            labels.append((f"高刷：来源标注{max(rates):g}Hz" +
                       ("，不符合" if max(rates) <= 60 and profile.high_refresh_required else ""))
                      if hits else "高刷：待核对")
        urls.extend(s.url for s, _ in hits)
    dimensions = EVIDENCE_PATTERNS
    for topic in profile.priorities:
        if topic not in dimensions:
            continue
        hits = topic_observations(phone_id,topic,sources)
        if hits:
            s, part = hits[0]
            label = "实测资料" if s.evidence_kind == "review" else "配置标注"
            labels.append(f"{topic}：{label} · {part.strip()[:90]}")
            urls.extend(s.url for s, _ in hits)
        else:
            labels.append(f"{topic}：待补" + ("实测" if topic in {"拍照", "续航"} else "配置或评测"))
    # Arbitrary requirements survive the conversation. Keyword overlap alone is
    # never promoted to a verified match; the answer can explain cited evidence.
    for requirement in profile.requirements:
        checked = numeric_requirement(requirement, sources)
        labels.append(f"具体要求：{requirement} · " + (checked[0] if checked else "待逐项核对"))
        if checked:
            urls.extend(checked[1])
    offers = {}
    for s in sources:
        if s.evidence_kind == "price":
            for offer in s.price_options:
                offers.setdefault(offer.variant, set()).add(offer.price)
    if any(len(values) > 1 for values in offers.values()):
        labels.append("价格：同版本来源冲突，待核对")
    return labels, list(dict.fromkeys(urls))


def missing_evidence(profile, selected, sources):
    gaps = []
    for pid in selected:
        labels, _ = card_matches(pid, profile, sources)
        if any("待" in label or "冲突" in label for label in labels) or not any(s.phone_id == pid and s.source_type != "snippet" for s in sources):
            gaps.append(pid)
    return gaps

"""Purchase constraints and evidence gaps, shared by cards and the tool-loop guard."""

import re
from .topics import EVIDENCE_PATTERNS

from .evidence_rules import photo_observation, endurance_observation, game_in_text, canonical_games, measured_game_parts, parameter_values, parameter_observations
from .selection_prices import minimum_storage, storage_gb, source_price_options, price_records, dated_promotion
from .selection_presentation import configuration_facts
from .feature_requirements import (RequirementCheck, feature_check, feature_values,
                                   feature_keys, literal_feature_updates, requirement_atoms, requirement_key)


def requested_count(text):
    match=re.search(r'(?:推荐|给我|列出|再找)\s*([一二两三四五1-5])\s*(?:款|台|部)',text)
    return (int(match[1]) if match[1].isdigit() else {'一':1,'二':2,'两':2,'三':3,'四':4,'五':5}[match[1]]) if match else None


def literal_requirements(profile, text):
    """Small unambiguous signals also work offline; no phone facts are inferred."""
    before = profile.model_dump()
    profile.games = canonical_games(profile.games)
    literal_feature_updates(profile, text)
    if re.search(r"预算|不超过|最多花|最高花|控制在|以内|取消预算|预算不限", text):
        profile.price_preference = "within_budget"
    if re.search(r"(?:想要|要买|想买|买|要).{0,8}\d{3,5}\s*(?:元|块|左右|价位).{0,5}(?:的)?手机|\d{3,5}\s*(?:元|块)?左右", text):
        profile.price_preference = "near_budget"
    count=requested_count(text)
    if count:profile.recommendation_count=count
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
    return bool(requirement_atoms(requirement)) and bool(feature_values(requirement, demand=True)
                or re.search(r"必须|务必|只接受|硬性|不能|不可|不得|需要|不要|排除", requirement)
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
        values=[(v,source.url) for v,u,r,source in parameter_observations(sources)
                if u==unit and (role is None or r==role)]
        if len({v for v, _ in values}) != 1:
            return None
        value = values[0][0]
        ok = value >= limits[0] if lower else value <= limits[0]
        return ("配置符合" if ok else "配置不符合"), [url for _, url in values]
    return None


def check_requirement(requirement, sources):
    parts=[a.text for a in requirement_atoms(requirement)]
    if len(parts)>1:
        checks=[check_requirement(part,sources) for part in parts]
        status=next((v for v in ('conflict','unsatisfied','unknown') if any(c.status==v for c in checks)),'satisfied')
        return RequirementCheck(status,tuple(dict.fromkeys(sid for c in checks for sid in c.source_ids)),
            tuple(dict.fromkeys(url for c in checks for url in c.urls)))
    if feature_keys(requirement):
        if re.search(r'\d+\s*(?:W|瓦)',requirement,re.I):
            return RequirementCheck('unknown')  # Presence does not prove a power threshold.
        return feature_check(requirement, sources)
    if requirement_key(requirement)=='storage':
        limit=storage_gb(requirement)
        values=[(storage_gb(o.variant),s) for s in sources for o in source_price_options(s)]
        if not limit or not values:return RequirementCheck('unknown')
        wanted=[v>=limit if re.search(r'至少|不低于|起码|>=|≥|以上',requirement) else v<=limit if re.search(r'最多|不超过|<=|≤|以下',requirement) else v==limit
                for v,s in values if v is not None]
        return RequirementCheck('satisfied' if any(wanted) else 'unsatisfied',
            tuple(dict.fromkeys(s.id for v,s in values)),tuple(dict.fromkeys(s.url for v,s in values)))
    numeric = numeric_requirement(requirement, sources)
    if numeric:
        return RequirementCheck('satisfied' if numeric[0]=='配置符合' else 'unsatisfied',
            source_ids=tuple(s.id for s in sources if numeric_requirement(requirement,[s])),
            urls=tuple(dict.fromkeys(numeric[1])))
    # Preserve conflicting numeric observations as a conflict, not a false match.
    singles=[numeric_requirement(requirement,[s]) for s in sources if s.source_type!='snippet']
    if any(singles):
        return RequirementCheck('conflict',tuple(s.id for s in sources if numeric_requirement(requirement,[s])),
            tuple(dict.fromkeys(url for hit in singles if hit for url in hit[1])))
    return RequirementCheck('unknown')


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
                observation=part.strip()
                if topic=='续航':
                    clauses=[p.strip() for p in re.split(r'[，,]',part) if p.strip()]
                    for i,clause in enumerate(clauses):
                        combined='，'.join(clauses[max(0,i-1):i+1])
                        if endurance_observation(combined):
                            marker=re.search(r'(?:经过|本次|在)[^，]{0,65}(?:测试|模型)|(?:\d+|[一二三四五六七八九十]+)小时[^，]{0,30}(?:测试|模型)',combined)
                            observation=combined[marker.start():] if marker else combined
                            break
                observations.append((source,observation))
    return observations


def recommendation_gaps(phone, profile, sources):
    """Unknown hard gates stay queryable leads, never qualified recommendations."""
    body=[s for s in sources if s.phone_id==phone.id and s.source_type!='snippet']
    gaps=[]
    if profile.budget_max:
        if phone.price is None:gaps.append('预算内版本价未确认')
        elif phone.price>profile.budget_max:gaps.append(f'已核{phone.variant}版本价{phone.price}元，超过预算{phone.price-profile.budget_max}元')
    prices=[s for s in body if s.evidence_kind=='price']
    usable_prices=[s for s in prices if source_price_options(s,phone.name)]
    # Price freshness is disclosed separately. A supported historical offer can
    # justify a provisional shortlist; it never proves current availability.
    if minimum_storage(profile) and (storage_gb(phone.variant) or 0)<minimum_storage(profile):gaps.append('存储版本未符合')
    labels,_=card_matches(phone.id,profile,body)
    if any('冲突' in label for label in labels):gaps.append('来源冲突未解决')
    if profile.high_refresh_required and not any(label.startswith('高刷：来源标注') and '不符合' not in label for label in labels):
        gaps.append('高刷硬要求未确认')
    for requirement in [a.text for r in profile.requirements for a in requirement_atoms(r)]:
        if not selection_threshold(requirement):continue
        if requirement_key(requirement)=='storage':
            # Capacity is checked against the selected, budget-compatible offer only.
            capacity=storage_gb(phone.variant); limit=storage_gb(requirement)
            if capacity is not None and limit is not None:
                atom=requirement_atoms(requirement)[0]
                ok=capacity>=limit if atom.operator=='min' else capacity<=limit if atom.operator=='max' else capacity==limit
                if not ok:gaps.append('硬要求不符合：'+requirement)
                continue
        checked=check_requirement(requirement,body)
        if checked.status!='satisfied':
            label={'unknown':'硬要求未确认','unsatisfied':'硬要求不符合','conflict':'硬要求来源冲突'}[checked.status]
            gaps.append(label+'：'+requirement)
    demands=[pair for requirement in profile.requirements for pair in feature_values(requirement,demand=True)]
    if any(len({v for k,v in demands if k==key})>1 for key,_ in demands):
        gaps.append('用户硬要求冲突，需澄清或取消其中一项')
    for topic in profile.priorities:
        if not configuration_facts(phone,[topic],body) and not topic_observations(phone.id,topic,body,profile.games):
            gaps.append(topic+'缺少相关依据')
    return gaps


def recommendation_notes(phone,profile,sources):
    prices=[s for s in sources if s.phone_id==phone.id and s.evidence_kind=='price' and source_price_options(s,phone.name)]
    notes=[]
    if prices and any(dated_promotion(s) for s in prices):notes.append('按历史报价初筛，当前可购价及优惠时效未核实')
    if prices and all(s.refresh_status in {'pending','failed','not_attempted','not_permitted'} for s in prices):notes.append('本轮未核到新报价，沿用历史价格作初步参考')
    return notes


def card_matches(phone_id, profile, sources,phone=None):
    sources = [s for s in sources if s.phone_id == phone_id and s.source_type != "snippet"]
    labels, urls = [], []
    if "游戏" in profile.priorities or profile.games:
        reviews = [s for s in sources if s.evidence_kind == "review" and any(
                   (any(game_in_text(game, part) for game in profile.games) if profile.games else
                    re.search(r"游戏|原神|星穹铁道|王者", part)) for part in measured_game_parts(s.quote))]
        facts=configuration_facts(phone,['游戏'],sources) if phone else []
        labels.append("游戏表现：有实测资料，见来源条件" if reviews else "游戏表现：配置支持初筛，实测表现待核实" if facts else "游戏表现：待补实测")
        urls.extend(s.url for s in reviews)
        urls.extend(s.url for s in sources if s.id in {sid for v,sid in facts})
        for game in profile.games:
            matched = [s for s in reviews if any(game_in_text(game, part) for part in measured_game_parts(s.quote))]
            labels.append(f"{game}：" + ("有测试记录" if matched else "待查"))
    if profile.high_refresh_required or "屏幕" in profile.priorities:
        hits = [(s,value) for value,unit,role,s in parameter_observations(sources)
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
            labels.append(f"{topic}：{label} · {part.strip()}")
            urls.extend(s.url for s, _ in hits)
        else:
            facts=configuration_facts(phone,[topic],sources,limit=1) if phone else []
            if facts:
                labels.append(topic+'：配置支持初步判断 · '+facts[0][0])
                urls.extend(s.url for s in sources if s.id==facts[0][1])
            else:labels.append(f"{topic}：待补" + ("实测" if topic in {"拍照", "续航"} else "配置或评测"))
    # Arbitrary requirements survive the conversation. Keyword overlap alone is
    # never promoted to a verified match; the answer can explain cited evidence.
    for requirement in profile.requirements:
        checked = check_requirement(requirement, sources)
        labels.append(f"具体要求：{requirement} · " + checked.label)
        urls.extend(checked.urls)
    offers = {}
    for s,offer in price_records(sources):
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

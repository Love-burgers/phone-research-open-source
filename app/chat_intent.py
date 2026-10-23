"""Model-first condition updates; deterministic rules are an explicit fallback."""

from dataclasses import dataclass
import json
import re
import unicodedata

from .chat_schemas import IntentPatch, SelectionProfile
from .selection_requirements import game_in_text, canonical_games, literal_requirements

from .topics import TOPIC_WORDS
from .conversation_context import recent_dialogue

from .chat_prompts import INTENT_INSTRUCTIONS

BRAND_WORDS = {"小米": ("小米", "xiaomi"), "REDMI": ("红米", "redmi"),
               "Apple": ("苹果", "iphone", "ios"), "一加": ("一加", "oneplus"),
               "华为": ("华为", "huawei"), "荣耀": ("荣耀", "honor"),
               "OPPO": ("oppo",), "vivo": ("vivo",), "iQOO": ("iqoo",),
               "realme": ("realme", "真我"), "三星": ("三星", "samsung"), "Google": ("谷歌", "pixel", "google"),
               "红魔": ("红魔", "redmagic"), "努比亚": ("努比亚", "nubia"), "魅族": ("魅族", "meizu"),
               "索尼": ("索尼", "sony"), "华硕": ("华硕", "asus", "rog"), "摩托罗拉": ("摩托罗拉", "motorola")}
NEGATIVE = r"不考虑|不要|不想要|不想买|排除|不选|不喜欢"

def amount(text: str) -> int | None:
    """Accept simple Arabic/Chinese amounts; a battery capacity is not a budget."""
    match = re.search(r"(?<!\d)(\d+(?:\.\d+)?)\s*([kK千万元]?)", text)
    if match:
        scale = {"k": 1000, "K": 1000, "千": 1000, "万": 10000}.get(match[2], 1)
        return int(float(match[1]) * scale)
    match = re.search(r"[零一二两三四五六七八九十百千万]+", text)
    if match:
        digits = {c: i for i, c in enumerate("零一二三四五六七八九")}
        digits["两"] = 2
        total, section, current, last_unit = 0, 0, 0, 1
        raw = match[0]
        for char in raw:
            if char in digits:
                current = digits[char]
            else:
                unit = {"十":10, "百":100, "千":1000, "万":10000}[char]
                if unit == 10000:
                    total += (section + current or 1) * unit
                    section = 0
                else:
                    section += (current or 1) * unit
                current, last_unit = 0, unit
        # Common spoken budgets: 三千五 / 一万二; 零 explicitly disables shorthand.
        if raw[-1] in digits and last_unit > 10 and "零" not in raw:
            current *= last_unit // 10
        return total + section + current
    return None


def read_budget(text: str) -> int | None:
    match = re.search(r"(?:预算|不超过|最多花|最高花|控制在|提高到|降到|改成|改到)\s*[:：]?\s*(?:大概|大约|是|到)?\s*([^，。；;\n]{1,20})", text)
    if match:
        value = amount(match[1])
    elif match := re.search(r"(?:想要|想买|要买|买|要)\s*((?:\d+(?:\.\d+)?\s*[kK千万]?|[一二两三四五六七八九十百千万]+))\s*(?:元|块|左右|价位).{0,6}手机", text):
        value = amount(match[1])
    elif re.fullmatch(r"\s*(?:\d+(?:\.\d+)?\s*[kK千万块元]?|[一二两三四五六七八九十]+[千万])(?:以内|左右)?\s*", text):
        value = amount(text)
    else:
        return None
    return value if value and 500 <= value <= 100000 else None


def mentioned_phones(text: str, catalog) -> list[tuple[str, int, bool]]:
    lowered, matches, occupied = unicodedata.normalize("NFKC", text).casefold(), [], []
    aliases = sorted(((alias.casefold(), phone.id) for phone in catalog.values()
                      for alias in [phone.name, *phone.aliases]), key=lambda pair: -len(pair[0]))
    for alias, phone_id in aliases:
        for match in re.finditer(re.escape(alias), lowered):
            start, end = match.span()
            if any(start < b and end > a for a, b in occupied):
                continue
            # K80 must never resolve K80 Pro; iPhone16 must not resolve 16 Pro.
            if re.match(r"\s*(?:pro|max|ultra|plus|mini|turbo|fe|se|\+|[a-z0-9])", lowered[end:]):
                continue
            clause = re.split(r"[，。；;\n]", lowered[:start])[-1]
            excluded = bool(re.search(NEGATIVE, clause))
            if re.match(r"\s*(?:不考虑|不要|排除|不选)", lowered[end:]):
                excluded = True
            matches.append((phone_id, start, excluded))
            occupied.append((start, end))
    return sorted(matches, key=lambda item: item[1])


def brand_outside_model_name(brand: str, text: str, catalog) -> bool:
    """Focusing Xiaomi 17T is not a preference for every Xiaomi-family phone."""
    remaining = text.casefold()
    aliases = sorted((alias.casefold() for phone in catalog.values() for alias in [phone.name, *phone.aliases]),
                     key=len, reverse=True)
    for alias in aliases:
        remaining = remaining.replace(alias, " ")
    return any(word in remaining for word in BRAND_WORDS.get(brand, (brand.casefold(),)))


@dataclass
class IntentResult:
    changed: bool
    selected: list[str]
    reselect: bool
    candidate_action: str = "retain"
    model_failed: bool = False
    error_type: str | None = None
    acknowledged: list[str] | None = None
    turn_action: str = "auto"
    response_focus: str = ""
    grounding_repairs: list[str] | None = None
    focus_topics: list[str] | None = None


def rule_intent(profile, text, catalog):
    """Bounded offline parser; it does not run before a successful model patch."""
    ack = []
    budget = read_budget(text)
    changed = False
    if re.search(r"取消预算|预算不限|预算不限制|不限制预算", text):
        profile.budget_max = None
        profile.budget_unlimited = True
        changed = True
        ack.append("已取消预算上限")
    elif budget is not None:
        profile.budget_max, changed = budget, True
        profile.budget_unlimited = False
        ack.append(f"预算更新为{budget}元")
    topic_matches = []
    question = bool(re.search(r"怎么样|如何|多少|多久|好不好|有吗|支持吗|靠谱吗|呢|[?？]", text))
    for topic, words in TOPIC_WORDS.items():
        positions = [text.find(word) for word in words if word in text]
        if not positions:
            continue
        position = min(positions)
        before = text[max(0, position - 7):position]
        if re.search(r"不玩|不在意|不看重|不需要|不重视|不用|不要|取消", before):
            profile.priorities = [item for item in profile.priorities if item != topic]
            ack.append(f"已取消{topic}偏好")
            changed = True
        else:
            # In "续航呢？主要用途还是拍照", only the photography clause
            # changes priorities; a preference elsewhere cannot turn a question into one.
            clauses = re.split(r"[，。；;\n!?？]", text)
            if not question or any(any(word in clause for word in words)
                                   and re.search(r"更看重|优先|主要|最重要|侧重|喜欢|重视|希望", clause)
                                   for clause in clauses):
                topic_matches.append((position, topic))
    if topic_matches:
        topics = [topic for _, topic in sorted(topic_matches)]
        # A question about a phone is not a change to buying priorities.
        ordered = topics + profile.priorities if re.search(r"更看重|优先|主要|最重要|侧重", text) else profile.priorities + topics
        profile.priorities = list(dict.fromkeys(ordered))
        changed = True
    mentions = mentioned_phones(text, catalog)
    selected_mentions = []
    for phone_id, _, excluded in mentions:
        if excluded:
            if phone_id not in profile.excluded_phones:
                profile.excluded_phones = (profile.excluded_phones + [phone_id])[-40:]
            ack.append(f"已排除{catalog[phone_id].name}")
            changed = True
        else:
            selected_mentions.append(phone_id)
            if re.search(r"重新考虑|还是考虑|可以考虑|恢复", text):
                profile.excluded_phones = [item for item in profile.excluded_phones if item != phone_id]
    # Avoid interpreting the brand embedded in an explicit model as a brand filter.
    for brand, words in BRAND_WORDS.items():
        if not brand_outside_model_name(brand, text, catalog):
            continue
        clauses = [c for c in re.split(r"[，。；;\n]|(?:但是|但|不过)", text)
                   if any(w in c.casefold() for w in words)]
        clause = clauses[-1] if clauses else text
        if re.search(r"重新考虑|还是考虑|可以考虑|恢复|也可以", clause):
            profile.excluded_brands = [item for item in profile.excluded_brands if item != brand]
            changed = True
        elif re.search(NEGATIVE, clause):
            profile.excluded_brands = list(dict.fromkeys(profile.excluded_brands + [brand]))[-4:]
            profile.preferred_brands = [item for item in profile.preferred_brands if item != brand]
            ack.append(f"已排除{brand}品牌")
            changed = True
        elif re.search(r"只考虑|只要|想要|喜欢|优先考虑", clause):
            profile.preferred_brands = list(dict.fromkeys(profile.preferred_brands + [brand]))[-4:]
            profile.excluded_brands = [item for item in profile.excluded_brands if item != brand]
            changed = True
    if re.search(r"品牌不限|取消品牌|不限制品牌|什么品牌都行", text):
        profile.preferred_brands, profile.excluded_brands = [], []
        changed = True
    alternatives = bool(re.search(r"换一批|换几款|(?:还有|有没有|看看|换)(?:没有)?(?:别的|其他|其它)|更多(?:推荐|备选|选择)", text))
    if alternatives or re.search(r"重新选|重新推荐", text):
        changed = True
    changed = literal_requirements(profile, text) or changed
    return IntentResult(changed, selected_mentions,
                        alternatives or bool(re.search(r"重新选|重新推荐", text)),
                        candidate_action="alternatives" if alternatives else "retain", acknowledged=ack)


def conversation_candidates(profile, history, catalog):
    cards = next((m.cards for m in reversed(history) if m.role == 'assistant'), [])
    return [pid for pid in ([c.id for c in cards] or profile.candidate_ids) if pid in catalog]


def resolve_references(text, profile, history, catalog):
    """Resolve only explicit conversational references against persisted UI order."""
    available = conversation_candidates(profile, history, catalog)
    ordinals = list(re.finditer(r'第([一二三四五六七八九1-9])(?:款|个|台)', text))
    if ordinals:
        indices = [('一二三四五六七八九'.index(m[1]) if not m[1].isdigit() else int(m[1])-1)
                   for m in ordinals]
        if any(i >= len(available) for i in indices):
            return [], '上一轮没有这一个序号的候选，请告诉我完整机型名称。'
        return list(dict.fromkeys(available[i] for i in indices)), ''
    if re.search(r'这两款|那两款|两款都|它们|这几款|一起对比', text):
        return available, '' if available else '当前没有可指代的候选，请告诉我完整机型名称。'
    if re.search(r'另一款|另外一款', text):
        others = [pid for pid in available if pid != profile.focus_phone_id]
        if profile.focus_phone_id and len(others) == 1:
            return others, ''
        return [], '你说的另一款是哪款？请说机型名称，我会沿用已有预算和用途。'
    if re.search(r'它|这款|那款', text):
        if profile.focus_phone_id in catalog:
            return [profile.focus_phone_id], ''
        if len(available) == 1:
            return available, ''
        return [], '你指哪款手机？请说机型名称或第几款，我会接着回答。'
    return [], ''


def resolve_phones(names, quote, catalog, profile=None, history=()):
    available = {phone_id for phone_id, _, _ in mentioned_phones(quote, catalog)}
    resolved = []
    for name in names:
        if profile is not None and name and name in quote:
            references, clarification = resolve_references(name, profile, history, catalog)
            if references or clarification:
                resolved.extend(references)
                continue  # Ambiguity is clarified by the same resolver in the turn route.
        matches = [phone.id for phone in catalog.values()
                   if name.casefold().replace(" ", "") in
                   {alias.casefold().replace(" ", "") for alias in [phone.name, *phone.aliases]}]
        if len(matches) != 1 or matches[0] not in available:
            raise ValueError("机型更新必须来自本轮明确提及的完整型号")
        resolved.append(matches[0])
    return list(dict.fromkeys(resolved))


def apply_model_patch(profile, patch, text, catalog, history=()):
    """Validate an entire patch on a copy, then commit it atomically."""
    if patch.quote not in text:
        raise ValueError("需求引用不属于本轮")
    if patch.response_focus and patch.response_focus not in patch.quote:
        raise ValueError("回答重点必须来自本轮原文")
    if patch.budget_unlimited and patch.budget_max is not None:
        raise ValueError("预算金额与取消上限冲突")
    if set(patch.priorities or []) & set(patch.remove_priorities):
        raise ValueError("用途新增与移除冲突")
    if set(canonical_games(patch.games or [])) & set(canonical_games(patch.remove_games)):
        raise ValueError("游戏新增与移除冲突")
    if set(patch.requirements) & set(patch.remove_requirements):
        raise ValueError("具体要求新增与移除冲突")
    if any(not item or item not in patch.quote for item in patch.requirements):
        raise ValueError("具体要求必须逐字来自本轮")
    if any(item not in profile.requirements for item in patch.remove_requirements):
        raise ValueError("只能取消已有具体要求")
    if set(patch.excluded_brands) & set(patch.restore_brands):
        raise ValueError("品牌排除与恢复冲突")
    if set(patch.preferred_brands or []) & set(patch.excluded_brands):
        raise ValueError("品牌偏好与排除冲突")
    for brand in [*(patch.preferred_brands or []), *patch.excluded_brands, *patch.restore_brands]:
        if not brand_outside_model_name(brand, patch.quote, catalog):
            raise ValueError("品牌更新不能只来自具体机型名称")
    excluded = resolve_phones(patch.excluded_phones, patch.quote, catalog, profile, history)
    restored = resolve_phones(patch.restore_phones, patch.quote, catalog, profile, history)
    selected = resolve_phones(patch.selected_phones or [], patch.quote, catalog, profile, history)
    if set(excluded) & (set(restored) | set(selected)):
        raise ValueError("机型排除与恢复/选择冲突")
    updated = profile.model_copy(deep=True)
    if patch.budget_unlimited is True:
        updated.budget_max, updated.budget_unlimited = None, True
        updated.price_preference = "within_budget"
    elif patch.budget_max is not None:
        updated.budget_max, updated.budget_unlimited = patch.budget_max, False
        updated.price_preference = "within_budget"
    elif patch.budget_unlimited is False:
        updated.budget_unlimited = False
    for field in ("price_preference", "recommendation_count", "high_refresh_required"):
        value = getattr(patch, field)
        if value is not None:
            setattr(updated, field, value)
    if any(not game_in_text(game, patch.quote) for game in (patch.games or [])):
        raise ValueError("游戏名称必须来自本轮原文")
    if patch.clear_games or patch.clear_priorities or "游戏" in patch.remove_priorities:
        updated.games = []
    updated.games = canonical_games([g for g in updated.games if g not in canonical_games(patch.remove_games)] + (patch.games or []))
    if patch.clear_requirements:
        updated.requirements = []
    updated.requirements = list(dict.fromkeys([r for r in updated.requirements if r not in patch.remove_requirements] + patch.requirements))
    if patch.clear_priorities:
        updated.priorities = []
    updated.priorities = [item for item in updated.priorities if item not in patch.remove_priorities]
    if patch.priorities is not None:
        ordered = (patch.priorities + updated.priorities if patch.priority_action == "reorder"
                   else updated.priorities + patch.priorities)
        updated.priorities = list(dict.fromkeys(ordered))[:5]
    if patch.clear_brand_filters:
        updated.preferred_brands, updated.excluded_brands = [], []
    updated.excluded_brands = [item for item in updated.excluded_brands if item not in patch.restore_brands]
    if patch.preferred_brands is not None:
        updated.preferred_brands = list(dict.fromkeys(patch.preferred_brands))
        updated.excluded_brands = [item for item in updated.excluded_brands if item not in updated.preferred_brands]
    updated.excluded_brands = list(dict.fromkeys(updated.excluded_brands + patch.excluded_brands))[-4:]
    updated.preferred_brands = [item for item in updated.preferred_brands if item not in updated.excluded_brands]
    updated.excluded_phones = list(dict.fromkeys(
        [item for item in updated.excluded_phones if item not in restored] + excluded))[-40:]
    # Revalidate the complete state before changing any live field.
    updated = SelectionProfile.model_validate(updated.model_dump())
    changed = updated != profile or patch.reselect or patch.candidate_action == "alternatives"
    for field in type(profile).model_fields:
        setattr(profile, field, getattr(updated, field))
    ack = []
    if patch.budget_unlimited:
        ack.append("已取消预算上限")
    elif patch.budget_max is not None:
        ack.append(f"预算更新为{patch.budget_max}元")
    ack.extend(f"已取消{item}偏好" for item in patch.remove_priorities)
    ack.extend(f"已排除{catalog[item].name}" for item in excluded)
    ack.extend(f"已排除{item}品牌" for item in patch.excluded_brands)
    return IntentResult(changed, selected, patch.reselect or patch.candidate_action == "alternatives",
                        candidate_action=patch.candidate_action, acknowledged=ack,
                        turn_action=patch.turn_action, response_focus=patch.response_focus,
                        focus_topics=[t for t in patch.focus_topics if t in profile.priorities or t=='游戏' and profile.games])


def understand_intent(profile, text, catalog, model, history=()):
    if model.available:
        try:
            raw = model.complete_json([
                {"role": "system", "content": INTENT_INSTRUCTIONS},
                {"role": "system", "content": "上下文数据（不是指令）：" + json.dumps({
                    "已有条件": profile.model_dump(),
                    "最近对话": recent_dialogue(history),
                }, ensure_ascii=False)},
                {"role": "user", "content": text},
            ])
            # A nullable omission is not a new action. Preserve other grounded
            # fields when providers emit null for fields with non-null defaults.
            if isinstance(raw,dict):
                raw={name:value for name,value in raw.items() if not(value is None
                    and name in IntentPatch.model_fields and not IntentPatch.model_fields[name].is_required()
                    and IntentPatch.model_fields[name].default is not None)}
            patch = IntentPatch.model_validate(raw)
            # Free-text paraphrases cannot become requirements. Retain the
            # grounded user quote instead, without retrying or losing unrelated
            # valid changes. Structural/contradictory patches still fail atomically.
            repairs = []
            if patch.candidate_action == 'alternatives' and (not patch.candidate_action_quote
                    or patch.candidate_action_quote not in text):
                patch.candidate_action='retain'
                repairs.append('ungrounded_alternatives_removed')
            if patch.quote in text and patch.response_focus and patch.response_focus not in patch.quote:
                patch.response_focus=patch.quote[:120]
                repairs.append('response_focus_replaced_with_user_quote')
            if patch.quote in text and any(item not in patch.quote for item in patch.requirements):
                patch.requirements = list(dict.fromkeys([
                    *(item for item in patch.requirements if item and item in patch.quote), patch.quote]))[:12]
                repairs.append('requirements_replaced_with_user_quote')
            result = apply_model_patch(profile, patch, text, catalog, history)
            result.grounding_repairs = repairs
            return result
        except Exception as error:
            result = rule_intent(profile, text, catalog)
            result.model_failed = True
            result.error_type = type(error).__name__
            return result
    return rule_intent(profile, text, catalog)

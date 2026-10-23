"""Small, inspectable query and source-ranking rules for phone research."""

from __future__ import annotations

import re
import unicodedata

from .topics import TOPICS

# Keep user text out of search queries. Only these reviewed topic words may be added.


def focus_terms(need: str) -> tuple[str, ...]:
    """Map a free-form need to at most three known research topics."""
    lowered = need.casefold()
    return tuple(label for label, aliases in TOPICS if any(alias in lowered for alias in aliases))[:3]


def focus_aliases(need: str) -> tuple[tuple[str, ...], ...]:
    """Use synonyms when checking content for the selected topics."""
    selected = set(focus_terms(need))
    return tuple(aliases for label, aliases in TOPICS if label in selected)


def build_queries(phone_name: str, need: str) -> tuple[str, str]:
    """Build the fallback queries from the requested topics."""
    review_terms = focus_terms(need) or ("续航", "拍照", "发热")
    return (
        f'"{phone_name}" 官方 参数 配置',
        f'"{phone_name}" 评测 {" ".join(review_terms)}',
    )


def source_score(phone_name: str, need: str, title: str, snippet: str) -> int:
    """Favor a matching model and relevant preview; this is not source truth scoring."""
    model_score = 6 if exact_model_in_text(phone_name, title) else 0
    model_score += 3 if exact_model_in_text(phone_name, snippet) else 0
    exact, longer_variant = _model_mentions(phone_name, f"{title} {snippet}")
    if exact and longer_variant:
        model_score -= 4
    aliases = focus_aliases(need)
    topic_score = sum(any(alias in title for alias in group) for group in aliases) * 3
    topic_score += sum(any(alias in snippet for alias in group) for group in aliases)
    return model_score + topic_score


def _canonical(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("+", "plus")
    return "".join(char for char in value if char.isalnum())


def _model_aliases(phone_name: str) -> tuple[str, ...]:
    canonical = _canonical(phone_name)
    aliases = {canonical}
    # Search engines use both the Chinese and international Xiaomi brand names.
    if canonical.startswith("小米"):
        aliases.add("xiaomi" + canonical.removeprefix("小米"))
    elif canonical.startswith("xiaomi"):
        aliases.add("小米" + canonical.removeprefix("xiaomi"))
    if canonical.startswith("红米"):
        aliases.add("redmi"+canonical.removeprefix("红米"))
    elif canonical.startswith("redmi"):
        aliases.add("红米" + canonical.removeprefix("redmi"))
    if canonical.startswith("realme"):
        aliases.add("真我" + canonical.removeprefix("realme"))
    elif canonical.startswith("真我"):
        aliases.add("realme" + canonical.removeprefix("真我"))
    return tuple(alias for alias in aliases if alias)


def _model_mentions(phone_name: str, value: str) -> tuple[bool, bool]:
    """Return (exact mention, longer variant mention) in normalized preview text."""
    text = unicodedata.normalize("NFKC", value).casefold().replace("+", "plus")
    exact = variant = False
    for alias in _model_aliases(phone_name):
        pattern = r"[\s\-_]*".join(re.escape(char) for char in alias)
        for match in re.finditer(pattern, text):
            before = text[match.start() - 1] if match.start() else ""
            if before and before.isascii() and before.isalnum():
                continue
            tail = text[match.end():]
            next_char = tail[:1]
            next_word = tail.lstrip(" -_/、·").split(maxsplit=1)[0] if tail.strip(" -_/、·") else ""
            is_variant = (
                bool(next_char and next_char.isascii() and next_char.isalnum())
                or bool(re.match(r"[a-z](?:\b|[\u4e00-\u9fff])", next_word))
                or any(next_word.startswith(suffix) for suffix in ("pro", "max", "plus", "ultra", "mini", "air", "fe", "卫星通信版", "卫星版", "satellite"))
            )
            if is_variant:
                variant = True
            else:
                exact = True
    return exact, variant


def exact_model_in_text(phone_name: str, value: str) -> bool:
    return _model_mentions(phone_name, value)[0]


def variant_only_title(phone_name: str, title: str) -> bool:
    """Exclude previews titled only for a longer model variant, e.g. 15 Pro vs 15."""
    exact, variant = _model_mentions(phone_name, title)
    return variant and not exact

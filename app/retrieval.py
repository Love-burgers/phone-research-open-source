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



# Re-export shared identities for existing callers.
from .phone_identity import (identity_key as _canonical, model_aliases as _model_aliases,
    model_mentions as _model_mentions, exact_model_in_text, variant_only_title)

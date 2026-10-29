"""Validated public conversation state; independent of the two-phone report API."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, HttpUrl


Topic = Literal["拍照", "续航", "游戏", "轻便", "屏幕"]
Brand = Literal["小米", "REDMI", "Apple", "一加", "华为", "荣耀", "OPPO", "vivo", "iQOO", "realme", "三星", "Google", "红魔", "努比亚", "魅族", "索尼", "华硕", "摩托罗拉"]
SESSION_PATTERN = r"^[0-9a-f]{32}$"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class SelectionProfile(StrictModel):
    budget_max: int | None = Field(default=None, ge=500, le=100000)
    budget_unlimited: bool = False
    price_preference: Literal["within_budget", "near_budget", "save_money"] = "within_budget"
    recommendation_count: int = Field(default=3, ge=1, le=5)
    high_refresh_required: bool = False
    games: list[str] = Field(default_factory=list, max_length=8)
    requirements: list[str] = Field(default_factory=list, max_length=12)
    answer_requirements: list[str] = Field(default_factory=list, max_length=8)
    priorities: list[Topic] = Field(default_factory=list, max_length=5)
    preferred_brands: list[Brand] = Field(default_factory=list, max_length=4)
    brand_scope: Literal['prefer','only'] = 'only'
    excluded_series: list[str] = Field(default_factory=list, max_length=12)
    excluded_brands: list[Brand] = Field(default_factory=list, max_length=4)
    excluded_phones: list[str] = Field(default_factory=list, max_length=40)
    candidate_ids: list[str] = Field(default_factory=list, max_length=5)
    explicit_candidate_ids: list[str] = Field(default_factory=list, max_length=5)
    focus_phone_id: str | None = None


class ChatRequest(StrictModel):
    message: str = Field(min_length=1, max_length=1500)
    session_id: str | None = Field(default=None, pattern=SESSION_PATTERN)
    search_mode: Literal["auto", "local", "web"] = "auto"


class PriceOption(StrictModel):
    variant: str = Field(min_length=1, max_length=80)
    price: int = Field(ge=500, le=100000)


class ChatSource(StrictModel):
    id: str
    phone_id: str
    title: str
    url: str
    quote: str
    checked_on: str
    source_type: Literal["knowledge", "page", "snippet"] = "knowledge"
    market: str = "中国大陆"
    published_on: str | None = None
    evidence_kind: Literal["spec", "review", "price", "unknown"] = "unknown"
    reference_price: int | None = Field(default=None, ge=500, le=100000)
    price_variant: str | None = None
    price_options: list[PriceOption] = Field(default_factory=list, max_length=12)
    refresh_status: Literal['not_requested','pending','verified','failed','not_permitted','not_attempted','superseded'] = 'not_requested'


class AnswerBlock(StrictModel):
    phone_id: str
    text: str = Field(min_length=1, max_length=600)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    background_ids: list[str] = Field(default_factory=list,max_length=4)


class PhoneCard(StrictModel):
    id: str
    name: str
    price: int | None
    variant: str
    price_url: str | None
    checked_on: str
    reason: str
    note: str
    matches: list[str] = Field(default_factory=list)
    match_sources: list[str] = Field(default_factory=list)


class PriceReference(AnswerBlock):
    name: str = Field(min_length=1,max_length=100)


class ChatMessage(StrictModel):
    role: Literal["user", "assistant"]
    content: str
    followup: str = Field(default="", max_length=240)
    blocks: list[AnswerBlock] = Field(default_factory=list)
    price_references: list[PriceReference] = Field(default_factory=list,max_length=3)
    sources: list[ChatSource] = Field(default_factory=list)
    cards: list[PhoneCard] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    mode: Literal["资料模式", "AI对话", "资料模式（模型暂不可用）", "资料模式（回答未通过核对）"] = "资料模式"
    agent_trace: dict | None = None
    details: list[str] = Field(default_factory=list)


class PhoneDetails(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9-]+$", max_length=64)
    name: str = Field(min_length=1, max_length=100)
    aliases: list[str] = Field(max_length=12)
    brand: Brand
    price: int | None = Field(default=None, ge=500, le=100000)
    variant: str
    price_url: HttpUrl | None = None
    checked_on: str
    topics: list[Topic] = Field(default_factory=list)
    note: str
    source_urls: list[HttpUrl] = Field(default_factory=list, max_length=8)
    price_options: list[PriceOption] = Field(default_factory=list, max_length=12)


class ChatSession(StrictModel):
    session_id: str = Field(pattern=SESSION_PATTERN)
    profile: SelectionProfile = Field(default_factory=SelectionProfile)
    messages: list[ChatMessage] = Field(default_factory=list)
    revision: int = 0
    discovered_phones: list[PhoneDetails] = Field(default_factory=list, max_length=20)


class IntentPatch(StrictModel):
    """A model patch must quote this turn, not silently replace the whole profile."""
    quote: str = Field(min_length=1, max_length=1500)
    turn_action: Literal["auto", "refine", "question", "revise", "recommend"] = "auto"
    response_focus: str = Field(default="", max_length=120)
    candidate_action_quote: str = Field(default="", max_length=120)
    focus_topics: list[Topic] = Field(default_factory=list, max_length=5)
    priority_action: Literal["add", "reorder"] = "add"
    budget_max: int | None = Field(default=None, ge=500, le=100000)
    budget_unlimited: bool | None = None
    price_preference: Literal["within_budget", "near_budget", "save_money"] | None = None
    recommendation_count: int | None = Field(default=None, ge=1, le=5)
    high_refresh_required: bool | None = None
    games: list[str] | None = Field(default=None, max_length=8)
    remove_games: list[str] = Field(default_factory=list, max_length=8)
    clear_games: bool = False
    requirements: list[str] = Field(default_factory=list, max_length=12)
    remove_requirements: list[str] = Field(default_factory=list, max_length=12)
    clear_requirements: bool = False
    priorities: list[Topic] | None = Field(default=None, max_length=5)
    remove_priorities: list[Topic] = Field(default_factory=list, max_length=5)
    clear_priorities: bool = False
    preferred_brands: list[Brand] | None = Field(default=None, max_length=4)
    excluded_brands: list[Brand] = Field(default_factory=list, max_length=4)
    restore_brands: list[Brand] = Field(default_factory=list, max_length=4)
    clear_brand_filters: bool = False
    excluded_phones: list[str] = Field(default_factory=list, max_length=5)
    restore_phones: list[str] = Field(default_factory=list, max_length=5)
    selected_phones: list[str] | None = Field(default=None, max_length=5)
    candidate_scope: Literal['retain', 'restrict', 'release'] = 'retain'
    candidate_scope_quote: str = Field(default='', max_length=1500)
    reselect: bool = False
    candidate_action: Literal["retain", "alternatives"] = "retain"


class ModelAnswer(StrictModel):
    blocks: list[AnswerBlock] = Field(min_length=1, max_length=8)

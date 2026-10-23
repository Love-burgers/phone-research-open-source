"""Verify patch application, preservation and fallback independently of LLM quality."""

import pytest

from app.chat_intent import understand_intent
from app.chat_schemas import SelectionProfile
from app.selection_catalog import load_catalog


class Model:
    available = True

    def __init__(self, patch):
        self.patch = patch
        self.calls = 0

    def complete_json(self, messages):
        self.calls += 1
        if isinstance(self.patch, Exception):
            raise self.patch
        return self.patch


@pytest.mark.parametrize("text,patch,expected", [
    ("别超过四千", {"budget_max": 4000}, {"budget_max": 4000}),
    ("手里就三千五", {"budget_max": 3500}, {"budget_max": 3500}),
    ("拍照我其实不太在意，主要打游戏", {"remove_priorities": ["拍照"], "priorities": ["游戏"], "priority_action": "reorder"},
     {"priorities": ["游戏", "续航"]}),
    ("苹果就算了", {"excluded_brands": ["Apple"]}, {"excluded_brands": ["Apple"]}),
    ("一直用华为，换机也想继续用", {"preferred_brands": ["华为"]}, {"preferred_brands": ["华为"]}),
    ("不看重续航了", {"remove_priorities": ["续航"]}, {"priorities": ["拍照"]}),
    ("取消预算上限", {"budget_unlimited": True}, {"budget_max": None, "budget_unlimited": True}),
    ("用途先都清空", {"clear_priorities": True}, {"priorities": []}),
    ("品牌不限", {"clear_brand_filters": True}, {"preferred_brands": [], "excluded_brands": []}),
    ("不考虑K80", {"excluded_phones": ["K80"]}, {"excluded_phones": ["redmi-k80"]}),
    ("重新考虑K80", {"restore_phones": ["K80"], "selected_phones": ["K80"]}, {"excluded_phones": []}),
    ("苹果也可以了", {"restore_brands": ["Apple"]}, {"excluded_brands": []}),
    ("电池容量多少？", {}, {"budget_max": 5000, "priorities": ["拍照", "续航"]}),
])
def test_model_updates_are_not_overridden_by_keyword_rules(text, patch, expected):
    profile = SelectionProfile(budget_max=5000, priorities=["拍照", "续航"])
    if "restore_phones" in patch:
        profile.excluded_phones = ["redmi-k80"]
    if "restore_brands" in patch:
        profile.excluded_brands = ["Apple"]
    model = Model({"quote": text, **patch})
    result = understand_intent(profile, text, load_catalog(), model)
    assert not result.model_failed
    assert model.calls == 1
    for key, value in expected.items():
        assert getattr(profile, key) == value
    if "budget_max" not in expected and "budget_unlimited" not in patch:
        assert profile.budget_max == 5000


@pytest.mark.parametrize("patch", [
    {"quote": "不是本轮", "budget_max": 9000},
    {"quote": "预算4000", "budget_max": 9000, "selected_phones": ["K80"]},
    {"quote": "预算4000", "budget_max": 4000, "budget_unlimited": True},
    {"quote": "预算4000", "priorities": ["游戏"], "remove_priorities": ["游戏"]},
    {"quote": "预算4000", "budget_max": -1},
    TimeoutError("private details"),
])
def test_invalid_patch_is_atomic_and_falls_back_without_retry(patch):
    profile = SelectionProfile(budget_max=5000, priorities=["拍照"])
    model = Model(patch)
    result = understand_intent(profile, "预算4000", load_catalog(), model)
    assert result.model_failed
    assert model.calls == 1
    assert profile.budget_max == 4000
    assert profile.priorities == ["拍照"]
    assert not profile.excluded_phones


def test_successful_noop_does_not_apply_rule_interpretation():
    profile = SelectionProfile(priorities=["续航"])
    result = understand_intent(profile, "拍照我其实不太在意", load_catalog(),
                               Model({"quote": "拍照我其实不太在意"}))
    assert not result.changed
    assert profile.priorities == ["续航"]


def test_unconfigured_model_is_not_called():
    model = Model(AssertionError("must not call"))
    model.available = False
    profile = SelectionProfile()
    result = understand_intent(profile, "预算4000，主要游戏", load_catalog(), model)
    assert not result.model_failed and model.calls == 0
    assert profile.budget_max == 4000 and profile.priorities == ["游戏"]

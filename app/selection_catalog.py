"""Small reviewed catalog. Topic tags shortlist hardware, not benchmark scores."""

import json
from pathlib import Path
from .chat_schemas import PhoneCard, PhoneDetails, SelectionProfile

from .selection_prices import select_offer, source_price_options, storage_gb, minimum_storage
from .phone_identity import series_excluded


DATA_DIR = Path(__file__).resolve().parent / "knowledge_samples"


class CatalogPhone(PhoneDetails):

    def card(self, profile: SelectionProfile, sources=()) -> PhoneCard:
        matched = [topic for topic in profile.priorities if topic in self.topics]
        reason = ("有可核对的" + "、".join(matched) + "相关配置，列为待选。" if matched
                  else "列入当前资料范围内的候选，可继续核对配置。")
        note = self.note
        from .selection_requirements import recommendation_notes
        notes=recommendation_notes(self,profile,sources)
        if notes:note+=' '+'；'.join(notes)+'。'
        price_records=[s for s in sources if s.phone_id==self.id and s.evidence_kind=='price']
        if price_records and all(s.refresh_status in {'pending','failed','not_attempted','not_permitted'} for s in price_records):
            note+=' 此为历史价格，本轮刷新尚未确认。'
        if self.price is None:
            note += " 当前价格未核对，不能确认是否符合预算。"
        elif profile.budget_max and self.price > profile.budget_max:
            note += " 页面参考价超出你的预算。"
        from .selection_requirements import card_matches
        matches, urls = card_matches(self.id, profile, sources,phone=self)
        if matches:
            reason = "；".join(matches[:2])
        from .selection_presentation import configuration_facts
        facts = configuration_facts(self, profile.priorities, sources, limit=3)
        if facts:
            reason = "配置：" + "、".join(value for value, _ in facts)
            fact_ids = {sid for _,sid in facts}
            urls = list(dict.fromkeys([*urls,*(s.url for s in sources if s.id in fact_ids)]))[:8]
        return PhoneCard(id=self.id, name=self.name, price=self.price, variant=self.variant,
                         price_url=str(self.price_url) if self.price_url else None, checked_on=self.checked_on,
                         reason=reason, note=note, matches=matches, match_sources=urls)


def load_catalog(path: Path | None = None) -> dict[str, CatalogPhone]:
    phones = [CatalogPhone.model_validate(item) for item in
              json.loads((path or DATA_DIR / "selection_catalog.json").read_text(encoding="utf-8"))]
    if len({phone.id for phone in phones}) != len(phones):
        raise ValueError("机型编号重复")
    return {phone.id: phone for phone in phones}


def eligible(phone: CatalogPhone, profile: SelectionProfile) -> bool:
    # Xiaomi/REDMI are treated together when the user excludes the Xiaomi family.
    return (phone.id not in profile.excluded_phones
            and phone.brand not in profile.excluded_brands
            and not (phone.brand == "REDMI" and "小米" in profile.excluded_brands)
            and not series_excluded(phone.name,profile.excluded_series)
            and (profile.brand_scope=='prefer' or not profile.preferred_brands or phone.brand in profile.preferred_brands
                 or phone.brand == "REDMI" and "小米" in profile.preferred_brands))


def select_variant(phone, profile):
    """Re-evaluate saved offers before filtering, even without another page fetch."""
    chosen = select_offer(phone.price_options, profile)
    if chosen:
        phone.price, phone.variant = chosen.price, chosen.variant


def variant_eligible(phone, profile):
    capacity = storage_gb(phone.variant)
    return not minimum_storage(profile) or capacity is None or capacity >= minimum_storage(profile)


def budget_position(price, profile):
    """Soft price-tier coverage, never a new lower-budget eligibility rule."""
    if not profile.budget_max or profile.price_preference == "save_money":
        return 0
    if price is None:
        return 1
    if price > profile.budget_max:
        return 3
    return 0 if price >= profile.budget_max * .8 else 2


def shortlist(catalog: dict[str, CatalogPhone], profile: SelectionProfile) -> list[str]:
    for phone in catalog.values():
        select_variant(phone, profile)
    options = [phone for phone in catalog.values() if eligible(phone, profile) and variant_eligible(phone, profile)
               and phone.price is not None and (profile.budget_unlimited
               or profile.budget_max is not None and phone.price <= profile.budget_max)]
    if profile.price_preference == "near_budget" and profile.budget_max:
        options = [phone for phone in options if phone.price >= profile.budget_max * .8]
    # A transparent feature match, never a claim about real photo/battery quality.
    def order(phone):
        matches = sum(len(profile.priorities) - i for i, topic in enumerate(profile.priorities)
                      if topic in phone.topics)
        return (bool(profile.preferred_brands and phone.brand not in profile.preferred_brands),bool(profile.priorities and profile.priorities[0] not in phone.topics),
                budget_position(phone.price,profile),-matches,
                phone.price if profile.price_preference == "save_money" else -phone.price, phone.id)
    return [phone.id for phone in sorted(options, key=order)[:profile.recommendation_count]]

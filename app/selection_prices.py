"""Literal offer validation and storage selection, without catalog/presentation dependencies."""

import re
from decimal import Decimal
from .chat_schemas import PriceOption


def _price_text(text):
    return re.sub(r'(?<=\d)\s*\.\s*(?=\d)', '.', text.replace('，',','))


def price_in_text(price, text, *, currency=False):
    """Match complete prices despite whitespace introduced inside decimal cents."""
    text=_price_text(text)
    suffix=r'\s*元' if currency else ''
    return bool(re.search(rf'(?<![\d.])(?:{price}|{price:,})(?:\.0+)?(?![\d.]){suffix}',text))

def select_offer(offers, profile):
    if not offers:
        return None
    minimum = minimum_storage(profile)
    sized = [o for o in offers if not minimum or (storage_gb(o.variant) or 0) >= minimum]
    allowed = [o for o in sized if not profile.budget_max or o.price <= profile.budget_max]
    choose = max if profile.price_preference == 'near_budget' and allowed else min
    return choose(allowed or sized or offers, key=lambda o: o.price)

def source_price_options(source):
    """Accept literal version/price records for both state and presentation."""
    if source.source_type == 'snippet' or source.evidence_kind != 'price' or source.market == '其他地区':
        return []
    options = source.price_options or ([PriceOption(price=source.reference_price,variant=source.price_variant)]
        if source.reference_price and source.price_variant else [])
    quote_variant = re.sub(r'\W','',source.quote).casefold()
    return [o for o in options if re.sub(r'\W','',o.variant).casefold() in quote_variant
            and price_in_text(o.price,source.quote)]


def unsupported_prices(text, sources, budget_max=None):
    """Distinguish cited prices from the user's budget and computed differences."""
    offers=[offer for source in sources for offer in source_price_options(source)]
    prices={offer.price for offer in offers}
    pairs={(re.sub(r'\s','',offer.variant).upper().replace('/','+'),offer.price) for offer in offers}
    differences={abs(price-budget_max) for price in prices} if budget_max else set()
    text=_price_text(text)
    issues=[]
    for match in re.finditer(r'(?<![\d.,])(\d+(?:,\d{3})*(?:\.\d+)?)\s*元',text):
        value=Decimal(match[1].replace(',',''))
        before=re.split(r'[，。；,;]',text[:match.start()])[-1][-12:]
        after=text[match.end():match.end()+8]
        if value==budget_max and (re.search(r'(?:预算|上限|最多)[^\d]{0,6}$',before) or re.match(r'预算|上限|限额',after)):continue
        if value in differences and re.search(r'(?:超预算|超出|便宜|省下|少花|差价|贵)[^\d]{0,5}$',before):continue
        variants=re.findall(r'\d{1,2}\s*GB\s*[+/]\s*(?:\d{2,4}\s*GB|[12]\s*TB)',text[max(0,match.start()-60):match.start()],re.I)
        variant=re.sub(r'\s','',variants[-1]).upper().replace('/','+') if variants else None
        if value in prices and (variant is None or (variant,value) in pairs):continue
        issues.append(match[0])
    return issues

def storage_gb(variant):
    capacities = re.findall(r"(\d+)\s*(GB|TB)", variant, re.I)
    return max((int(n) * (1024 if unit.upper() == "TB" else 1) for n, unit in capacities), default=None)

def minimum_storage(profile):
    return max((storage_gb(r) or 0 for r in profile.requirements
        if re.search(r"至少|不低于|起码|>=|≥", r) and (storage_gb(r) or 0) >= 128), default=0)

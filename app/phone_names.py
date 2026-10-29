"""Identify explicitly written phone names without a five-model allowlist."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
import unicodedata

from .chat_schemas import Brand
from .selection_catalog import CatalogPhone


from .phone_identity import BRANDS, PHONE_PATTERN, BARE_REDMI, SUFFIX_PATTERN, normalized_name


def checked_today():
    return datetime.now(timezone(timedelta(hours=8))).date().isoformat()


@dataclass(frozen=True)
class PhoneMention:
    name: str
    brand: Brand
    original: str
    start: int


def parse_phone_names(text: str) -> list[PhoneMention]:
    result = []
    for match in PHONE_PATTERN.finditer(unicodedata.normalize("NFKC", text)):
        if re.match(r"(?:Hz|GB|TB|mAh|mm|元|万像素|克)(?![a-z])", text[match.end():], re.I):
            continue  # Brand followed by a measurement isn't a model number.
        brand, prefix = BRANDS[match["brand"].casefold()]
        family = (match["family"] or "").replace(" ", "")
        if family == "游戏手机":
            family = "phone"
        family = family.upper() if len(family) <= 2 else family.title()
        model = family + match["number"] + (match["letter"] or "").upper()
        raw = match["suffix"].strip().casefold()
        suffix_words = re.findall(SUFFIX_PATTERN, raw)
        suffix = " ".join("Pro Max" if word.replace(" ", "") == "promax" else word.title()
                          for word in suffix_words)
        name = f"{prefix} {model}" + (f" {suffix}" if suffix else "")
        result.append(PhoneMention(name, brand, match[0], match.start()))
    for match in BARE_REDMI.finditer(text):
        if any(item.start <= match.start() < item.start + len(item.original) for item in result):
            continue
        parsed = parse_phone_names("REDMI " + match[0])[0]
        result.append(PhoneMention(parsed.name, parsed.brand, match[0], match.start()))
    return sorted(result, key=lambda item: item.start)


def register_phone(mention: PhoneMention, catalog: dict[str, CatalogPhone]) -> CatalogPhone:
    wanted = normalized_name(mention.name)
    for phone in catalog.values():
        if wanted in {normalized_name(phone.name), *(normalized_name(alias) for alias in phone.aliases)}:
            return phone
    phone_id = "phone-" + hashlib.sha256(wanted.encode()).hexdigest()[:20]
    aliases = [mention.name, mention.original, mention.name.replace(" ", "")]
    for alias, (brand, prefix) in BRANDS.items():
        if brand == mention.brand:
            aliases.append(alias + mention.name.removeprefix(prefix).replace(" ", ""))
    aliases = list(dict.fromkeys(aliases))[:12]
    # Registration records a user-specified name, not proof of existence or price.
    return CatalogPhone(id=phone_id, name=mention.name, aliases=aliases, brand=mention.brand,
                        variant="版本与售价待核对", checked_on=checked_today(),
                        note="你指定的机型，将查询公开资料；未确认售价与可购买性。")

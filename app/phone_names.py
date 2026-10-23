"""Identify explicitly written phone names without a five-model allowlist."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import re
import unicodedata

from .chat_schemas import Brand
from .selection_catalog import CatalogPhone


BRANDS = {
    "iphone": ("Apple", "iPhone"), "苹果": ("Apple", "iPhone"),
    "xiaomi": ("小米", "Xiaomi"), "小米": ("小米", "Xiaomi"),
    "redmi": ("REDMI", "REDMI"), "红米": ("REDMI", "REDMI"),
    "oneplus": ("一加", "一加"), "一加": ("一加", "一加"),
    "华为": ("华为", "华为"), "huawei": ("华为", "华为"),
    "荣耀": ("荣耀", "荣耀"), "honor": ("荣耀", "荣耀"),
    "oppo": ("OPPO", "OPPO"), "vivo": ("vivo", "vivo"),
    "iqoo": ("iQOO", "iQOO"), "realme": ("realme", "realme"), "真我": ("realme", "realme"),
    "三星": ("三星", "三星"), "samsung": ("三星", "三星"),
    "pixel": ("Google", "Pixel"), "谷歌": ("Google", "Pixel"),
    "红魔": ("红魔", "红魔"), "redmagic": ("红魔", "红魔"),
    "努比亚": ("努比亚", "努比亚"), "nubia": ("努比亚", "努比亚"),
    "魅族": ("魅族", "魅族"), "meizu": ("魅族", "魅族"),
    "索尼": ("索尼", "索尼"), "sony": ("索尼", "索尼"),
    "华硕": ("华硕", "ROG"), "asus": ("华硕", "ROG"), "rog": ("华硕", "ROG"),
    "摩托罗拉": ("摩托罗拉", "摩托罗拉"), "motorola": ("摩托罗拉", "摩托罗拉"),
}
PHONE_PATTERN = re.compile(
    r"(?<![a-z0-9])(?P<brand>" + "|".join(BRANDS) + r")\s*"
    r"(?P<family>mate|nova|pura|magic|note|reno|ace|neo|gt|phone|游戏手机|xperia|edge|razr|find\s*[nx]|[acnksvtyxpz])?\s*"
    r"(?P<number>\d{1,3})(?:\s*(?P<letter>[a-z](?![a-z])))?"
    r"(?P<suffix>(?:\s*(?:pro\s*max|promax|pro|ultra|plus|max|mini|fe|se|turbo|\+|标准版|至尊版|青春版))*)",
    re.IGNORECASE,
)
BARE_REDMI = re.compile(r"(?<![a-z0-9])k\d{2,3}(?:\s*(?:pro|ultra|至尊版))*", re.IGNORECASE)


def normalized_name(text):
    return "".join(c for c in unicodedata.normalize("NFKC", text).casefold().replace("+", "plus") if c.isalnum())


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
        suffix_words = re.findall(r"pro\s*max|promax|pro|ultra|plus|max|mini|fe|se|turbo|\+|至尊版|青春版", raw)
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

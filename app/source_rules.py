"""Pure public-source identity rules; importing this module performs no IO."""

from urllib.parse import urlsplit

OFFICIAL_DOMAINS = {"Apple": "apple.com.cn", "小米": "mi.com", "REDMI": "mi.com",
                    "一加": "oneplus.com/cn", "华为": "consumer.huawei.com/cn",
                    "荣耀": "honor.com/cn", "OPPO": "oppo.com/cn", "vivo": "vivo.com.cn",
                    "iQOO": "iqoo.com", "realme": "realme.com/cn", "三星": "samsung.com/cn",
                    "红魔": "redmagic.com", "努比亚": "nubia.com", "魅族": "meizu.com",
                    "索尼": "sonystyle.com.cn", "华硕": "asus.com.cn", "摩托罗拉": "moto.com.cn"}

def official_phone_url(brand, url):
    host = (urlsplit(url).hostname or "").lower()
    domains = [OFFICIAL_DOMAINS.get(brand, "").split('/')[0]]
    if brand == "Apple":
        domains.append("apple.com")
    if brand == "iQOO":
        domains.append("vivo.com.cn")
    return any(domain and (host == domain or host.endswith('.' + domain)) for domain in domains)

"""Shared literal model identity and suffix boundaries; no catalog or I/O."""
import re
import unicodedata
from functools import lru_cache

BRANDS = {
    'iphone':('Apple','iPhone'),'苹果':('Apple','iPhone'),
    'xiaomi':('小米','Xiaomi'),'小米':('小米','Xiaomi'),
    'redmi':('REDMI','REDMI'),'红米':('REDMI','REDMI'),
    'oneplus':('一加','一加'),'一加':('一加','一加'),
    '华为':('华为','华为'),'huawei':('华为','华为'),
    '荣耀':('荣耀','荣耀'),'honor':('荣耀','荣耀'),
    'oppo':('OPPO','OPPO'),'vivo':('vivo','vivo'),'iqoo':('iQOO','iQOO'),
    'realme':('realme','realme'),'真我':('realme','realme'),
    '三星':('三星','三星'),'samsung':('三星','三星'),
    'pixel':('Google','Pixel'),'谷歌':('Google','Pixel'),
    '红魔':('红魔','红魔'),'redmagic':('红魔','红魔'),
    '努比亚':('努比亚','努比亚'),'nubia':('努比亚','努比亚'),
    '魅族':('魅族','魅族'),'meizu':('魅族','魅族'),
    '索尼':('索尼','索尼'),'sony':('索尼','索尼'),
    '华硕':('华硕','ROG'),'asus':('华硕','ROG'),'rog':('华硕','ROG'),
    '摩托罗拉':('摩托罗拉','摩托罗拉'),'motorola':('摩托罗拉','摩托罗拉'),
}
# Chinese editions remain literal identities; we do not assume every maker's
# 至尊版 is a global Ultra, or infer regional equivalence from marketing names.
SUFFIX_PATTERN = (r'(?:pro\s*max|promax|pro\s*plus|pro|ultra|plus|max|mini|air|fe|se|turbo|satellite)(?![a-z])|\+'
    r'|至尊纪念版|至尊版|青春版|竞速版|领先版|至臻版|活力版|典藏版|旗舰版|卫星通信版|卫星版')
STANDARD_PATTERN = r'标准版|标准款|standard'
PHONE_PATTERN = re.compile(
    r'(?<![a-z0-9])(?P<brand>'+ '|'.join(BRANDS)+r')\s*'
    r'(?P<family>mate|nova|pura|magic|note|reno|ace|neo|gt|phone|游戏手机|xperia|edge|razr|find\s*[nx]|[acnksvtyxpz])?\s*'
    r'(?P<number>\d{1,3})(?:\s*(?P<letter>[a-z](?![a-z])))?'
    r'(?P<suffix>(?:\s*(?:'+SUFFIX_PATTERN+'|'+STANDARD_PATTERN+r'))*)',re.I)
BARE_REDMI = re.compile(r'(?<![a-z0-9])k\d{2,3}(?:\s*(?:'+SUFFIX_PATTERN+'|'+STANDARD_PATTERN+r'))*',re.I)


def normalized_name(value):
    return ''.join(c for c in unicodedata.normalize('NFKC',value).casefold().replace('+','plus') if c.isalnum())


def model_spans(text):
    spans=[(m.start(),m.end(),identity_key(m[0])) for m in PHONE_PATTERN.finditer(text)]
    spans.extend((m.start(),m.end(),identity_key('REDMI '+m[0])) for m in BARE_REDMI.finditer(text)
                 if not any(a<=m.start()<b for a,b,k in spans))
    return sorted(spans)


def owner_at(text,position,default=None):
    boundary=max(text.rfind(' / ',0,position),text.rfind('\n\n',0,position))
    names=[n for n in model_spans(text) if boundary<n[0]<position]
    return names[-1][2] if names else identity_key(default) if default else None


@lru_cache(maxsize=4096)
def identity_key(value):
    key=normalized_name(value)
    for alias,(_,canonical) in sorted(BRANDS.items(),key=lambda pair:-len(pair[0])):
        if key.startswith(alias):
            key=normalized_name(canonical)+key[len(alias):]
            break
    return re.sub(r'(?:标准版|标准款|standard)$','',key)


def series_key(name):
    match=PHONE_PATTERN.search(name)
    if not match:return None
    return '|'.join([BRANDS[match['brand'].casefold()][0],(match['family'] or '').casefold().replace(' ',''),match['number']])


def series_excluded(name,series):
    return series_key(name) in series


def model_aliases(name):
    key=identity_key(name); aliases={key}
    for _,(_,canonical) in BRANDS.items():
        prefix=normalized_name(canonical)
        if key.startswith(prefix):
            aliases.update(alias+key[len(prefix):] for alias,(_,target) in BRANDS.items()
                           if normalized_name(target)==prefix)
            break
    return tuple(sorted(aliases,key=lambda s:(-len(s),s)))


def variant_tail(tail):
    tail=unicodedata.normalize('NFKC',tail).casefold().lstrip(' -_/、·')
    tail=re.sub(r'^(?:'+STANDARD_PATTERN+r')','',tail).lstrip()
    return bool(re.match(r'(?:'+SUFFIX_PATTERN+r')',tail,re.I)
        or re.match(r'[a-z](?:\b|[\u4e00-\u9fff])',tail))


def model_mentions(name,value):
    text=unicodedata.normalize('NFKC',value).casefold().replace('+','plus')
    exact=variant=False
    for alias in model_aliases(name):
        pattern=r'[\s\-_]*'.join(re.escape(c) for c in alias)
        for match in re.finditer(pattern,text):
            before=text[match.start()-1] if match.start() else ''
            if before and before.isascii() and before.isalnum():continue
            tail=text[match.end():]
            next_char=tail[:1]
            if (next_char and next_char.isascii() and next_char.isalnum()) or variant_tail(tail):
                variant=True
            else:exact=True
    return exact,variant


def exact_model_in_text(name,value):
    return model_mentions(name,value)[0]


def variant_only_title(name,title):
    exact,variant=model_mentions(name,title)
    return variant and not exact


def source_identity_matches(name,title,quote):
    # Anonymous trusted chunks are permitted. Explicitly different variants in
    # a title or sole subject in the body must not be rebound by phone_id alone.
    wanted=identity_key(name)
    for text in (title,quote):
        names={identity_key(m[0]) for m in PHONE_PATTERN.finditer(text)
               if not re.match(r'(?:Hz|GB|TB|mAh|mm|元|万像素|克)(?![a-z])',text[m.end():],re.I)}
        if names and wanted not in names:return False
    return not variant_only_title(name,title) and not variant_only_title(name,quote)

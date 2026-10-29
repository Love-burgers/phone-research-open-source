"""Literal offer validation and storage selection, without catalog/presentation dependencies."""

import re
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from .chat_schemas import PriceOption, PriceReference
from .feature_requirements import requirement_atoms
from .phone_identity import identity_key, model_spans


VARIANT_PATTERN=r'(?:\d{1,2}\s*(?:GB)?\s*[+/]\s*)?(?:\d{2,4}\s*GB|[12]\s*TB)'
MONEY_PATTERN=re.compile(r'(?<![\d.])(?<!\d,)(?:(?P<currency>RMB|人民币|[¥￥])\s*)?(?P<value>\d+(?:,\d{3})*(?:\.\d+)?)(?(currency)\s*(?:元)?|\s*元)(?![a-zA-Z\d.])',re.I)


def variant_key(value):
    value=re.sub(r'\s','',value).upper().replace('/','+')
    return re.sub(r'^(\d{1,2})\+',r'\1GB+',value)


def price_subject(source):
    title={key for a,b,key in model_spans(source.title)}
    if len(title)==1:return next(iter(title))
    body={key for a,b,key in model_spans(source.quote)}
    return next(iter(body)) if len(body)==1 else None


def price_facts(text, subject=None):
    """Each literal amount belongs to its local subject, never the whole page.

    Explicit following attribution (3000元是X的价格) takes precedence. Shared
    multi-model labels remain ambiguous instead of guessing a respectively map.
    Anonymous rows may use the caller's independently established product scope.
    """
    text=_price_text(text);names=model_spans(text)
    default=identity_key(subject) if subject else None
    previous=0
    for match in MONEY_PATTERN.finditer(text):
        boundary=max(text.rfind('\n',0,match.start()),text.rfind(' / ',0,match.start()))
        prior=[n for n in names if boundary<n[0]<match.start()]
        owner=prior[-1][2] if prior else default
        if len(prior)>1 and re.fullmatch(r'[\s、和与及/]+',text[prior[-2][1]:prior[-1][0]]) and re.search(r'分别|各|均|都',text[prior[-1][1]:match.start()]):owner=None
        tail=re.split(r'[，,。；;\n]',text[match.end():])[0]
        attribution=re.match(r'\s*(?:是|为|属于|对应)\s*(?:的)?',tail)
        if attribution:
            following=model_spans(tail[attribution.end():])
            if following and following[0][0]<4:owner=following[0][2]
        denied=None
        denial=re.match(r'\s*(?:不是|不属于|不能(?:当作|作为|算作|视为))\s*',tail)
        if denial:
            following=model_spans(tail[denial.end():])
            if following and following[0][0]<4:denied=following[0][2]
        negative=bool(re.search(r'(?:不是|并非|不为)\s*$',text[max(previous,match.start()-8):match.start()]))
        if negative:denied=owner
        start=max(previous,boundary+1)
        prefix=text[start:match.start()]
        versions=list(re.finditer(VARIANT_PATTERN,prefix,re.I))
        variant=variant_key(versions[-1][0]) if versions and len(prefix)-versions[-1].end()<48 else None
        if versions:
            earlier=[n for n in names if boundary<n[0]<start+versions[-1].start()]
            version_owner=earlier[-1][2] if earlier else default
            if version_owner!=owner and not attribution:variant=None
        following=re.match(r'\s*(?:对应|版本为|的版本是)\s*('+VARIANT_PATTERN+')',text[match.end():],re.I)
        if following:variant=variant_key(following[1])
        yield {'owner':owner,'denied_owner':denied,'negative':negative or bool(denied),'variant':variant,'price':Decimal(match['value'].replace(',','')),'match':match}
        previous=match.end()


def source_price_facts(source, phone_name=None):
    if source.source_type=='snippet' or source.evidence_kind!='price' or source.market=='其他地区' or source.refresh_status=='superseded':return []
    # Generated archive headers are not independent evidence for their raw body.
    raw=re.split(r' / (?=本文属于原创|如若转载|转载请|版权声明|本文链接|原文链接)',source.quote.split('原文：',1)[-1])[0]
    return [f for f in price_facts(raw,phone_name or price_subject(source)) if not f['negative'] and price_amount_role(raw,f)=='price']


def price_amount_role(text, fact):
    """Separate a handset price from subsidy caps, savings and other amounts."""
    prefix=re.split(r'[，,。；;！!\n]',_price_text(text)[:fact['match'].start()])[-1]
    roles=list(re.finditer(r'到手价|实付|成交价|售价|券后价|补贴后|优惠后|最高补贴|最多补贴|补贴额|补贴|立减|减免|直降|优惠额|省下|省|节省|价值|礼盒|赠品|运费|定金',prefix))
    if roles and roles[-1][0] in {'到手价','实付','成交价','售价','券后价','补贴后','优惠后'}:return 'price'
    if re.search(r'(?:预算|上限|差价|(?:最高|最多)?补贴|补贴额|立减|减免|直降|优惠额|省下|省|便宜|节省|价值|礼盒|赠品|运费|定金)[^，。；\d]{0,8}$',prefix):
        return 'adjustment'
    return 'price'


def conditional_price(text, fact):
    """A coupon/subsidy/trade-in amount is not an unconditional offer."""
    prefix=re.split(r'[，,。；;！!\n]',text[:fact['match'].start()])[-1]
    return bool(re.search(r'叠加|满减|领券|券后|补贴后|国补|以旧换新|(?:会员|学生|新用户|新客)(?:专享|价)|优惠后|(?:需|须|限)[^，。；]{0,12}(?:券|会员|补贴|换新)',prefix))


def dated_promotion(source):
    """Reading an old deal page today does not renew the deal."""
    return bool(source.published_on and source.published_on < source.checked_on
        and re.search(r'活动|促销|限时|叠加|满减|领券|券后|补贴|国补|以旧换新|优惠后',source.quote))


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

def source_price_options(source, phone_name=None):
    """Accept literal version/price records for both state and presentation."""
    if source.source_type == 'snippet' or source.evidence_kind != 'price' or source.market == '其他地区' or source.refresh_status=='superseded':
        return []
    options = source.price_options or ([PriceOption(price=source.reference_price,variant=source.price_variant)]
        if source.reference_price and source.price_variant else [])
    subject=identity_key(phone_name) if phone_name else price_subject(source)
    facts=source_price_facts(source,phone_name)
    raw=_price_text(source.quote.split('原文：',1)[-1])
    return [o for o in options if any(f['price']==o.price and f['variant']==variant_key(o.variant)
            and f['owner']==subject and not conditional_price(raw,f) for f in facts)]


def price_lookup_notes(phone, sources):
    """Explain what was found without upgrading leads or deals to current offers."""
    own=[s for s in sources if s.phone_id==phone.id]
    body=[s for s in own if s.source_type!='snippet']
    notes=[]
    for source in body:
        options=source_price_options(source,phone.name)
        if options:
            date=source.published_on or source.checked_on
            notes.append(date+'的'+('历史促销记录' if dated_promotion(source) else '版本报价')+'：'+
                '、'.join(f'{o.variant} {o.price}元' for o in options[:3]))
        text=_price_text(source.quote.split('原文：',1)[-1])
        facts=[f for f in (source_price_facts(source,phone.name) if source.evidence_kind=='price' else price_facts(text,phone.name)) if f['owner']==identity_key(phone.name) and not f['negative'] and price_amount_role(text,f)=='price']
        conditional=[f for f in facts if conditional_price(text,f)]
        if conditional:
            notes.append('另有优惠条件价'+ '、'.join(f"{f['price']}元" for f in conditional[:3])+'，优惠资格和时效未确认')
        if facts and source.market=='地区未确认':notes.append('正文含报价，但适用地区未确认')
        elif facts and not options and not conditional:notes.append('正文含报价，但版本对应关系尚未确认')
    if not notes:
        notes.append('仅找到搜索摘要报价，尚未取得对应版本报价正文' if any(
            MONEY_PATTERN.search(s.quote+' '+s.title) for s in own if s.source_type=='snippet')
            else '已取得配置资料，本轮尚未取得版本报价正文')
    return list(dict.fromkeys(notes))


def historical_price_references(catalog, sources):
    """Cited historical prices are references, separate from qualified candidates."""
    references=[]
    for pid in dict.fromkeys(s.phone_id for s in sources):
        if pid not in catalog:continue
        phone=catalog[pid]
        own=[s for s in sources if s.phone_id==pid and s.source_type!='snippet' and s.evidence_kind=='price'
             and s.market!='其他地区' and s.refresh_status!='superseded']
        if any(source_price_options(s,phone.name) and fresh_price_source(s) for s in own):continue
        entries=[];ids=[];seen=set()
        for source in own:
            raw=_price_text(source.quote.split('原文：',1)[-1])
            previous=None
            for fact in source_price_facts(source,phone.name):
                if fact['owner']!=identity_key(phone.name) or not 500<=fact['price']<=100000:continue
                prefix=re.split(r'[，,。；;！!\n]',raw[:fact['match'].start()])[-1]
                if re.search(r'(?:预算|上限|差价|立减|优惠额|补贴额|减免)[^，。；]{0,6}$',prefix):continue
                variant=fact['variant']
                if (not variant and previous and previous['variant'] and previous['owner']==fact['owner']
                    and not re.search(r'[。；;\n]| / ',raw[previous['match'].end():fact['match'].start()])
                    and not model_spans(raw[previous['match'].end():fact['match'].start()])
                    and fact['match'].start()-previous['match'].end()<100):
                    variant=previous['variant']
                previous={**fact,'variant':variant}
                conditional=conditional_price(raw,fact)
                key=(variant,fact['price'],conditional,source.published_on or source.checked_on)
                if key in seen:continue
                if not variant and any(v and (price,condition,date)==key[1:] for v,price,condition,date in seen):continue
                seen.add(key)
                date=(source.published_on+'发布') if source.published_on else source.checked_on+'核对，原发布日期未记录'
                channel=next((name for name in ['京东国际','天猫国际','京东','天猫','官网'] if name in raw),'渠道未注明')
                detail=f"{date}，{channel}：{variant or '版本未注明'} {'优惠条件价' if conditional else '报价'}{fact['price']:f}元"
                if conditional:detail+='（'+prefix.strip()[-60:]+'）'
                if source.market=='地区未确认':detail+='，适用地区未确认'
                if len('；'.join([*entries,detail]))>450:continue
                entries.append(detail);ids.append(source.id)
                if len(entries)>=3:break
            if len(entries)>=3:break
        if entries:
            references.append(PriceReference(phone_id=pid,name=phone.name,text='历史价格参考：'+'；'.join(entries)+
                '。当前价格及优惠是否有效未核实，不能据此确认今天在预算内。',evidence_ids=list(dict.fromkeys(ids))))
        if len(references)>=3:break
    return references


def sanitize_price_source(source,phone_name):
    if source.evidence_kind!='price':return source
    offers=source_price_options(source,phone_name)
    scalar=next((o for o in offers if o.price==source.reference_price and o.variant==source.price_variant),None)
    return source.model_copy(update={'price_options':offers,'reference_price':scalar.price if scalar else None,
        'price_variant':scalar.variant if scalar else None})


def sanitize_session_prices(session):
    """Repair current derived state lazily, while keeping original messages intact."""
    latest={}
    for message in reversed(session.messages):
        if message.role=='assistant':
            for source in message.sources:latest.setdefault(source.id,source)
    for phone in session.discovered_phones:
        records=[s for s in latest.values() if s.phone_id==phone.id and s.evidence_kind=='price'
                 and str(phone.price_url or '')==s.url]
        if not records:continue
        original=[o for s in records for o in (s.price_options or ([PriceOption(price=s.reference_price,variant=s.price_variant)] if s.reference_price and s.price_variant else []))]
        valid=[o for s in records for o in source_price_options(s,phone.name)]
        if not original or all(o in valid for o in original):continue
        phone.price_options=list({(o.variant,o.price):o for o in valid}.values())
        if not any(o.price==phone.price and o.variant==phone.variant for o in valid):
            phone.price=None;phone.price_url=None;phone.variant='版本与售价待核对'
            phone.note='此前派生报价未通过机型归属核对，当前报价已撤回；历史原文与回答仍保留。'
    return session


def price_records(sources):
    """Current facts per source/version; keep opposing URLs for conflict checks."""
    records=[(s,o) for s in sources for o in source_price_options(s)]
    active={(s.phone_id,variant_key(o.variant)) for s,o in records if not dated_promotion(s)}
    records=[(s,o) for s,o in records if not dated_promotion(s) or (s.phone_id,variant_key(o.variant)) not in active]
    newest={}
    for s,o in records:
        key=(s.phone_id,s.url,s.market,o.variant)
        newest[key]=max(newest.get(key,''),s.checked_on)
    return [(s,o) for s,o in records if s.checked_on==newest[(s.phone_id,s.url,s.market,o.variant)]]


def price_claim_role(text, match):
    """Bind temporal assertions to each price, ignoring negated freshness labels."""
    # Keep list headings (e.g. 当前价格如下：) across version separators, but
    # reset at paragraph/sentence boundaries. Later temporal labels take over.
    prefix=re.split(r'[。；;\n]',text[:match.start()])[-1]
    prefix=re.sub(r'(?:不是|并非|不能(?:当作|作为|视为)?|不代表|非|未(?:查到|查得|核实到|确认)|没有(?:查到|确认))[^，。；;\n]{0,8}(?:最新|今天|当前|目前|现价)[^，。；;\n]{0,8}', '',prefix)
    events=list(re.finditer(r'历史|原(?:记录|日期)|当时|最新|(?:今天|当前|目前)(?!预算|要求|条件|问题|推荐)|现价',prefix))
    current=bool(events and re.match(r'最新|今天|当前|目前|现价',events[-1][0]))
    # A suffix such as “3000元是今天的到手价” is an assertion too.
    suffix=re.split(r'[，,。；;\n]',text[match.end():])[0]
    if re.match(r'\s*[，,]?\s*(?:是|为|即|属于)(?:今天|最新|当前|目前|现价)',text[match.end():]):current=True
    deal=current and bool(re.search(r'到手价|成交价|实付',prefix+suffix))
    return current,deal


def fresh_price_source(source, *, deal=False):
    today=datetime.now(timezone(timedelta(hours=8))).date().isoformat()
    if source.source_type!='page' or source.checked_on!=today or dated_promotion(source) or source.refresh_status in {'pending','failed','not_attempted','not_permitted','superseded'}:return False
    return not deal or bool(re.search(r'到手价|成交价|实付',source.quote))


def reported_official_prices(text,sources,phone_name):
    """Allow a cited article's attributed report, without calling it an official source."""
    if not re.search(r'原文|正文|文章|报道|转述|来源(?:提到|记载|记录|称)',text):return False
    claims=[]
    for f in price_facts(text,phone_name):
        prefix=re.split(r'[，,。；;\n]',text[:f['match'].start()])[-1]
        if re.search(r'官网|官方(?:售价|价|建议零售价)',prefix):claims.append(f)
    if not claims:return False
    for claim in claims:
        supported=False
        for source in sources:
            raw=_price_text(source.quote.split('原文：',1)[-1])
            for fact in source_price_facts(source,phone_name):
                prefix=re.split(r'[，,。；;\n]',raw[:fact['match'].start()])[-1]
                if (fact['owner']==claim['owner'] and fact['price']==claim['price']
                    and (not claim['variant'] or fact['variant']==claim['variant']) and re.search(r'官网|官方(?:售价|价|建议零售价)',prefix)):
                    supported=True
        if not supported:return False
    return True


def unsupported_prices(text, sources, budget_max=None, *, phone_name=None):
    """Distinguish cited prices from the user's budget and computed differences."""
    offers=[offer for source in sources for offer in source_price_options(source)]
    prices={offer.price for offer in offers}
    differences={abs(price-budget_max) for price in prices} if budget_max else set()
    text=_price_text(text)
    issues=[]
    subject=phone_name or next((price_subject(s) for s in sources if price_subject(s)),None)
    facts=[(s,f) for s in sources for f in source_price_facts(s,phone_name)]
    owned_prices=prices|{f['price'] for s,f in facts if f['owner']==(identity_key(subject) if subject else None)}
    differences={abs(price-budget_max) for price in owned_prices} if budget_max else set()
    for claim in price_facts(text,subject):
        match=claim['match']
        value=Decimal(match['value'].replace(',',''))
        before=re.split(r'[，。；,;]',text[:match.start()])[-1][-12:]
        after=text[match.end():match.end()+8].lstrip()
        if value==budget_max and (re.search(r'(?:预算|上限|最多)[^\d]{0,6}$',before) or re.match(r'预算|上限|限额|(?:以内|以下|之内|内)',after)):continue
        if value in differences and re.search(r'(?:超预算|超出|便宜|省下|少花|差价|贵)[^\d]{0,5}$',before):continue
        if claim['negative']:
            if any(f['price']==value and f['owner'] and f['owner']!=claim['denied_owner'] for s,f in facts):continue
            issues.append(match[0]);continue
        variant=claim['variant']
        current,deal=price_claim_role(text,match)
        supported=[(s,f) for s,f in facts if f['price']==value and f['owner']==claim['owner']
                   and (variant is None or f['variant']==variant)]
        if supported and (not current or any(fresh_price_source(s,deal=deal) for s,o in supported)):continue
        issues.append(match[0])
    return issues

def storage_gb(variant):
    capacities = re.findall(r"(\d+)\s*(GB|TB)", variant, re.I)
    return max((int(n) * (1024 if unit.upper() == "TB" else 1) for n, unit in capacities), default=None)

def minimum_storage(profile):
    return max((storage_gb(r) or 0 for item in profile.requirements for r in [a.text for a in requirement_atoms(item) if a.key=='storage']
        if re.search(r"至少|不低于|起码|>=|≥", r) and (storage_gb(r) or 0) >= 128), default=0)

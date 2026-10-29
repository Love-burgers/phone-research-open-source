"""Admit observed official configuration/price bodies, not search previews."""
import re
from .knowledge_models import KnowledgeDocument
from .source_rules import official_phone_url
from .retrieval import variant_only_title
from .selection_prices import source_price_options


def source_documents(sources, catalog):
    documents = []
    for source in sources:
        phone = catalog.get(source.phone_id)
        if (not phone or source.source_type != "page" or source.market == "其他地区"
            or source.refresh_status in {'pending','failed','not_attempted','not_permitted','superseded'}
            or not official_phone_url(phone.brand,source.url) or source.evidence_kind not in {"spec","price"}):
            continue
        if variant_only_title(phone.name,source.title) or ('satellite' in source.url.lower()
                and not re.search(r'卫星|satellite',phone.name,re.I)):
            continue
        data = dict(phone_name=phone.name,title=source.title,source_url=source.url,
            checked_on=source.checked_on,published_on=source.published_on,market=source.market,
            evidence_kind=source.evidence_kind)
        if source.evidence_kind == "spec":
            if re.search(r"\d+(?:\.\d+)?\s*(?:万|亿|mAh|Hz|W|mm|英寸|GB)",source.quote,re.I):
                documents.append(KnowledgeDocument(**data,text=source.quote))
        elif source.market == "中国大陆":
            for offer in source_price_options(source,phone.name):
                documents.append(KnowledgeDocument(**data,reference_price=offer.price,price_variant=offer.variant,
                    text=f"{phone.name}，版本{offer.variant}，页面展示价{offer.price}元。核对日{source.checked_on}。原文：{source.quote}"))
    return documents

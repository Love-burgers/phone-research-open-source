"""Offline coverage/admission audit; never a semantic accuracy or human review score."""
import argparse
from collections import Counter
import json
from pathlib import Path

from app.chat_schemas import ChatSource
from app.knowledge import KnowledgeBase, load_index
from app.selection_prices import source_price_options
from app.retrieval import variant_only_title


def audit(index):
    issues=[];models={}
    for c in index.chunks:
        row=models.setdefault(c.phone_name,{'kinds':Counter(),'source_urls':set(),'checked_on':set()})
        row['kinds'][c.evidence_kind]+=1;row['source_urls'].add(c.source_url);row['checked_on'].add(c.checked_on)
        reasons=[]
        if variant_only_title(c.phone_name,c.title):reasons.append('possible_variant_mismatch')
        if c.evidence_kind=='price':
            source=ChatSource(id=c.chunk_id,phone_id='audit',title=c.title,url=c.source_url,quote=c.text,
                checked_on=c.checked_on or '2000-01-01',market=c.market,evidence_kind='price',
                reference_price=c.reference_price,price_variant=c.price_variant)
            if not source_price_options(source):reasons.append('price_variant_not_literally_supported')
        if not c.checked_on:reasons.append('missing_checked_on')
        if reasons:issues.append({'chunk_id':c.chunk_id,'phone_name':c.phone_name,'reasons':reasons,'url':c.source_url})
    return {'coverage':KnowledgeBase(index).coverage(),'admission_issues':issues,
        'models':{name:{'kinds':dict(row['kinds']),'source_urls':sorted(row['source_urls']),
                        'checked_on':sorted(d for d in row['checked_on'] if d)} for name,row in models.items()},
        'human_review_status':'pending','scope':'Structural admission checks and coverage, not truth/availability/semantic accuracy.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(load_index(args.index))
    with args.output.open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
    print(json.dumps({'coverage':result['coverage'],'admission_issue_count':len(result['admission_issues'])}))


if __name__=='__main__':main()

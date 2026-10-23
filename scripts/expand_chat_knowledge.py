"""Bounded official-page import, following observed links; no chat API calls."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from app.knowledge import KnowledgeBase, KnowledgeIndex, build_index, load_index, save_index
from app.knowledge_ingest import source_documents
from app.phone_names import parse_phone_names, normalized_name, register_phone
from app.phone_sources import PublicPhoneSources, official_phone_url
from app.selection_catalog import load_catalog
from app.selection_presentation import configuration_facts
from app.tools import fetch_html


def observed_links(html, url, brand):
    soup = BeautifulSoup(html,'html.parser')
    found = []
    prefixes = {'一加':'一加','Apple':'iPhone','OPPO':'OPPO','vivo':'vivo','iQOO':'iQOO'}
    for anchor in soup.select('a[href]'):
        target=urljoin(url,anchor['href']).split('#')[0].split('?')[0]
        if not official_phone_url(brand,target) or not re.search(r'/vivo/|/smartphones/|/cn/[^/]+/(?:specs)?$',urlsplit(target).path):
            continue
        label=anchor.get_text(' ',strip=True) or anchor.get('title','') or anchor.get('aria-label','')
        if not label:
            label=' '.join(image.get('alt','') for image in anchor.select('img[alt]'))
        names=parse_phone_names(label)
        if not names:
            names=parse_phone_names(prefixes.get(brand,brand)+' '+label)
        if len(names)!=1 or names[0].brand not in {brand,'iQOO' if brand=='vivo' else brand}:
            continue
        if re.search(r'耳机|平板|手表|配件|手环|充电器|360|°',label):
            continue
        if re.search(r'卫星通信版|卫星版|satellite',label+' '+target,re.I):
            continue  # This importer only identifies currently supported full names.
        found.append((names[0].name,target))
    return found


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-pages',type=int,default=100,choices=range(1,101))
    parser.add_argument('--target-models',type=int,default=100)
    parser.add_argument('--seeds',type=Path,help='Observed name/URL pairs; directory entries use a brand name.')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    pages=args.output/'pages';pages.mkdir()
    original=load_index(args.base)
    catalog=load_catalog()
    # Seed URLs come from the existing, separately preserved source registry.
    seed_names=['vivo X200','OPPO Findx9','一加 13']
    queue=[]
    for name in seed_names:
        row=next(c for c in original.chunks if normalized_name(c.phone_name)==normalized_name(name))
        queue.append((name,row.source_url))
    if args.seeds:
        queue=[(row['phone_name'],row['url']) for row in json.loads(args.seeds.read_text(encoding='utf-8'))]
    visited=set();attempted=set();admitted={};events=[]
    def read(item):
        name,url=item
        try:
            html=fetch_html(url)
            key=hashlib.sha256(url.encode()).hexdigest()[:24]
            (pages/(key+'.html')).write_text(html,encoding='utf-8')
            return name,url,html,None,key
        except Exception as error:
            return name,url,None,type(error).__name__,None
    with ThreadPoolExecutor(max_workers=4) as pool:
        while queue and len(visited)<args.max_pages:
            # Prefer specification links, then unseen models, keeping duplicate
            # product pages from consuming all of the acquisition allowance.
            queue.sort(key=lambda item:(normalized_name(item[0]) in attempted,
                not bool(re.search(r'param|specs',item[1])),item[0]))
            batch=[]
            while queue and len(batch)<4 and len(visited)<args.max_pages:
                item=queue.pop(0)
                if item[1] in visited:continue
                visited.add(item[1]);attempted.add(normalized_name(item[0]));batch.append(item)
            for name,url,html,error,key in pool.map(read,batch):
                event=dict(phone_name=name,url=url,error_type=error,snapshot=key)
                if html:
                    mentions=parse_phone_names(name)
                    if len(mentions)!=1:
                        if name in {'vivo','OPPO','一加'}:
                            for item in observed_links(html,url,name):
                                if item[1] not in visited and item not in queue:queue.append(item)
                        events.append(event)
                        continue
                    phone=register_phone(mentions[0],catalog);catalog[phone.id]=phone
                    loader=PublicPhoneSources(args.output/'cache',fetcher=lambda _:html)
                    result=loader.fetch_result(phone,url,'配置 芯片处理器 屏幕刷新率 相机主摄 电池容量 充电 重量 存储版本价格')
                    docs=source_documents(result['sources'],catalog)
                    # Require at least a literal parameter, not a marketing/menu
                    # mention. Existing exact-model extraction also applies.
                    specs=[d for d in docs if d.evidence_kind=='spec' and configuration_facts(phone,
                        ['游戏','拍照','续航','轻便','屏幕'],result['sources'])]
                    accepted=[*specs,*(d for d in docs if d.evidence_kind=='price')]
                    for doc in accepted:
                        admitted[(doc.phone_name,doc.source_url,doc.evidence_kind,doc.price_variant)]=doc
                    event['admitted_documents']=len(accepted)
                    for item in observed_links(html,url,phone.brand):
                        if item[1] not in visited and item not in queue:queue.append(item)
                    for link in result['candidates']:
                        if re.search(r'param|specs',link['url']) and link['url'] not in visited:
                            queue.append((phone.name,link['url']))
                events.append(event)
                (args.output/'acquisition.json').write_text(json.dumps(events,ensure_ascii=False,indent=2),encoding='utf-8')
            total={normalized_name(c.phone_name) for c in original.chunks}|{normalized_name(d.phone_name) for d in admitted.values()}
            print(json.dumps(dict(pages=len(visited),models=len(total),documents=len(admitted))),flush=True)
            if len(total)>=args.target_models:break
    corpus=args.output/'new-corpus.jsonl'
    corpus.write_text('\n'.join(d.model_dump_json() for d in admitted.values())+'\n',encoding='utf-8')
    incoming=build_index(corpus) if admitted else None
    base=KnowledgeBase.from_path(args.base)
    seen={c.chunk_id for c in original.chunks}
    additions=[c for c in incoming.chunks if c.chunk_id not in seen] if incoming else []
    vectors=(base.embeddings.embed_documents([c.text for c in additions])
             if additions and original.method=='vector' else [None]*len(additions))
    updated=KnowledgeIndex(method=original.method,embedding_signature=original.embedding_signature,
        chunks=[*original.chunks,*(c.model_copy(update={'vector':v}) for c,v in zip(additions,vectors))])
    save_index(updated,args.output/('index-vector.json' if original.method=='vector' else 'index-bm25.json'))
    manifest=dict(models=len({normalized_name(c.phone_name) for c in updated.chunks}),chunks=len(updated.chunks),
        kinds=dict(Counter(c.evidence_kind for c in updated.chunks)),pages_attempted=len(visited),
        max_pages=args.max_pages,added_chunks=len(additions),chat_api_requests=0,embedding_api_requests=0,
        original_index=str(args.base),original_sha256=hashlib.sha256(args.base.read_bytes()).hexdigest(),
        limitation='Official parameters and observed prices only; model count is not independent review coverage or current availability.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':main()

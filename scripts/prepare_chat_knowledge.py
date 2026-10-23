"""Build a new production corpus from preserved official snapshots and observed pages."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3

from app.chat_schemas import ChatSource
from app.knowledge import KnowledgeDocument,build_index,save_index,load_index
from app.knowledge_ingest import source_documents
from app.local_embeddings import LocalEmbeddings
from app.phone_names import parse_phone_names,register_phone
from app.selection_catalog import load_catalog,DATA_DIR,CatalogPhone


def prepare(output):
    output.mkdir(parents=True,exist_ok=False)
    paths=[Path('runs/rag-benchmark-20261021-v2/corpus.jsonl'),
           DATA_DIR/'selection_phones.jsonl',DATA_DIR/'selection_reviews.jsonl']
    docs=[]
    for path in paths:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():continue
            raw=json.loads(line)
            mentions=parse_phone_names(raw['phone_name'])
            if len(mentions)==1:raw['phone_name']=mentions[0].name
            if '中国大陆' in raw.get('market',''):raw['market']='中国大陆'
            docs.append(KnowledgeDocument.model_validate(raw))
    captured={}
    def collect(source,catalog):
        for doc in source_documents([source],catalog):
            key=(doc.phone_name,doc.source_url,doc.evidence_kind,doc.price_variant)
            prior=captured.get(key)
            if not prior or ((doc.checked_on or ''),len(doc.text)) > ((prior.checked_on or ''),len(prior.text)):
                captured[key]=doc
    with sqlite3.connect('runs/chat-sessions.sqlite3') as db:
        for (raw,) in db.execute('select data from sessions'):
            session=json.loads(raw);catalog=load_catalog()
            for phone in session.get('discovered_phones',[]):
                catalog[phone['id']]=CatalogPhone.model_validate(phone)
            for msg in session['messages']:
                for raw_source in msg.get('sources',[]):
                    source=ChatSource.model_validate(raw_source)
                    collect(source,catalog)
    for path in Path('runs').glob('**/phone-source-cache/*.json'):
        try:
            cached=json.loads(path.read_text(encoding='utf-8'))
            rows=cached.get('sources',[]) if isinstance(cached,dict) else cached
            for row in rows:
                source=ChatSource.model_validate(row);catalog=load_catalog()
                if source.phone_id not in catalog:
                    mentions=parse_phone_names(source.title)
                    if len(mentions)!=1:continue
                    phone=register_phone(mentions[0],catalog)
                    if phone.id!=source.phone_id:continue
                    catalog[phone.id]=phone
                collect(source,catalog)
        except (ValueError,TypeError,KeyError):
            continue
    docs.extend(captured.values())
    corpus=output/'corpus.jsonl'
    corpus.write_text('\n'.join(d.model_dump_json() for d in docs)+'\n',encoding='utf-8')
    base=build_index(corpus)
    embeddings=LocalEmbeddings('runs/model-cache')
    previous=load_index(Path('runs/rag-comparison-20261021-v2/index-vector.json'))
    cached={c.text:c.vector for c in previous.chunks} if previous.embedding_signature==embeddings.signature else {}
    missing=[c.text for c in base.chunks if c.text not in cached]
    cached.update(zip(missing,embeddings.embed_documents(missing)))
    index=base.model_copy(update={'method':'vector','embedding_signature':embeddings.signature,
        'chunks':[c.model_copy(update={'vector':cached[c.text]}) for c in base.chunks]})
    save_index(index,output/'index-vector.json')
    report={'models':len({c.phone_name for c in index.chunks}),'chunks':len(index.chunks),
        'documents':len(docs),'kinds':dict(Counter(c.evidence_kind for c in index.chunks)),
        'new_local_document_embeddings':len(missing),'embedding_api_requests':0,
        'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
        'snapshot_scope':'Preserved official specs and observed official bodies; original reviews remain only where separately verified. Not live stock or full-market coverage.'}
    (output/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    prepare(parser.parse_args().output)

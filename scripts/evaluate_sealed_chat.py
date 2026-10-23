"""Seal external/developer cases before a bounded live run; never claim blindness."""
import argparse
import hashlib
import json
import re
import shutil
from importlib import metadata
from pathlib import Path

from app.chat import SelectionAssistant
from app.chat_intent import understand_intent
from app.chat_schemas import ChatMessage, ChatRequest, SelectionProfile, ChatSession
from app.chat_store import ChatStore
from app.conversation_policy import route_turn
from app.selection_catalog import load_catalog
from app.config import get_settings
from app.knowledge import KnowledgeBase
from scripts.verify_game_followup import BoundedModel


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint():
    root=Path(__file__).resolve().parents[1]
    paths=[*sorted((root/'app').rglob('*.py')),*sorted((root/'scripts').glob('*.py')),
           root/'app/knowledge_samples/selection_index.json',root/'app/knowledge_samples/selection_catalog.json',
           root/'app/knowledge_samples/public_source_directory.jsonl',root/'pyproject.toml']
    settings=get_settings()
    # Compare behavior-affecting settings and installed versions without storing
    # credentials, endpoints or private paths in the plan.
    runtime={'settings':settings.model_dump(mode='json',exclude={
        'model_api_key','deepseek_api_key','embedding_api_key','chat_access_token'}),
        'dependencies':sorted((d.metadata['Name'],d.version) for d in metadata.distributions() if d.metadata['Name']),
        'index_sha256':digest(Path(settings.chat_knowledge_index_path)) if settings.chat_knowledge_index_path else None}
    return hashlib.sha256((''.join(str(p.relative_to(root))+digest(p) for p in paths)+json.dumps(runtime,sort_keys=True)).encode()).hexdigest()


def seal(cases,output,limit,origin):
    if not 1 <= limit <= 36:
        raise ValueError('Provider request limit must be between 1 and 36')
    data=json.loads(cases.read_text(encoding='utf-8'))
    if not isinstance(data,list) or not data or len(data)>20:
        raise ValueError('Expected 1-20 cases')
    ids=[]
    for case in data:
        ids.append(case['id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',case['id']):raise ValueError('Unsafe case ID')
        if case['kind']=='intent':
            if not isinstance(case['input'],str) or not case['input']:raise ValueError('Empty input')
        elif case['kind']=='conversation':
            if not 1<=len(case['turns'])<=8:raise ValueError('Invalid turn count')
            for turn in case['turns']:ChatRequest.model_validate(turn)
        else:raise ValueError('Unknown case kind')
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate case IDs')
    output.mkdir(parents=True,exist_ok=False)
    settings=get_settings()
    index=Path(settings.chat_knowledge_index_path) if settings.chat_knowledge_index_path else Path(__file__).resolve().parents[1]/'app/knowledge_samples/selection_index.json'
    shutil.copy2(index,output/'knowledge-input.json')
    seeds={}
    for case in data:
        if case.get('seed_session'):
            seed=Path(case['seed_session'])
            ChatSession.model_validate_json(seed.read_text(encoding='utf-8'))
            target=output/(case['id']+'-seed.json');shutil.copy2(seed,target)
            seeds[target.name]=digest(target)
    plan={'schema_version':1,'cases':str(cases.resolve()),'cases_sha256':digest(cases),
          'knowledge_input_sha256':digest(output/'knowledge-input.json'),
          'source_fingerprint':source_fingerprint(),'max_provider_requests':limit,
          'origin_declared':origin,'blindness':'not_independently_verified','human_review_status':'pending','seed_sha256':seeds}
    (output/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    return plan


def run(output):
    plan=json.loads((output/'plan.json').read_text(encoding='utf-8'))
    cases=Path(plan['cases'])
    if digest(cases)!=plan['cases_sha256'] or source_fingerprint()!=plan['source_fingerprint']:
        raise ValueError('Cases or implementation changed after sealing; preserve this plan and create a new one')
    if digest(output/'knowledge-input.json')!=plan['knowledge_input_sha256']:raise ValueError('Knowledge snapshot changed')
    if any(digest(output/name)!=sha for name,sha in plan.get('seed_sha256',{}).items()):raise ValueError('Seed snapshot changed')
    # Exclusive marker prevents a failed or interrupted paid run being repeated.
    (output/'run.started').open('x').close()
    model=BoundedModel(output,limit=plan['max_provider_requests'])
    if not model.available:raise ValueError('No configured model')
    catalog=load_catalog();ids=list(catalog)[:3];rows=[]
    for case in json.loads(cases.read_text(encoding='utf-8')):
        if len(model.requests)>=model.limit:
            rows.append({'id':case['id'],'status':'skipped_request_limit'});continue
        if case['kind']=='intent':
            profile=SelectionProfile(candidate_ids=ids,**case['profile'])
            before=profile.model_dump()
            history=[ChatMessage(role='assistant',content='当前候选已列出。',followup=case.get('followup',''),
                      cards=[catalog[pid].card(profile) for pid in ids])]
            intent=understand_intent(profile,case['input'],catalog,model,history)
            route=route_turn(before,profile,intent,has_candidates=True,selected=[])
            rows.append({'id':case['id'],'kind':'intent','input':case['input'],'before':before,'after':profile.model_dump(),
                'intent':vars(intent),'route':vars(route),'expected_route':case.get('expected_route'),
                'route_matches_declared_expectation':route.operation==case.get('expected_route')})
        else:
            # Use a fresh session in the ordinary store, available to the UI.
            knowledge_path=output/(case['id']+'-index.json')
            shutil.copy2(output/'knowledge-input.json',knowledge_path)
            knowledge=KnowledgeBase.from_path(knowledge_path,search_method=get_settings().chat_retrieval_method)
            service=SelectionAssistant(ChatStore(Path('runs/chat-sessions.sqlite3')),model_client=model,knowledge=knowledge)
            sid=None;turns=[]
            if case.get('seed_session'):
                seed=ChatSession.model_validate_json((output/(case['id']+'-seed.json')).read_text(encoding='utf-8'))
                fresh=service.store.create()
                clone=seed.model_copy(deep=True,update={'session_id':fresh.session_id,'revision':fresh.revision})
                service.store.save(clone);sid=clone.session_id
            for request in case['turns']:
                remaining=model.limit-len(model.requests)
                if remaining<2:
                    turns.append({'input':request['message'],'status':'skipped_request_limit'});break
                if sid and case.get('reload_between_turns',False):
                    # Discard in-memory assistant/catalog state. The next turn
                    # must recover the real saved session through ChatStore.
                    service=SelectionAssistant(ChatStore(Path('runs/chat-sessions.sqlite3')),model_client=model,knowledge=knowledge)
                service.agent_policy=service.agent_policy.model_copy(update={'max_model_calls':min(6,remaining)})
                session=service.turn(ChatRequest(**request,session_id=sid));sid=session.session_id
                turns.append({'input':request,'result':session.model_dump(mode='json')})
                (output/(case['id']+'-turns.json')).write_text(json.dumps(turns,ensure_ascii=False,indent=2),encoding='utf-8')
            rows.append({'id':case['id'],'kind':'conversation','session_id':sid,'turns':turns})
        (output/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    report={**plan,'actual_provider_requests':sum(r.get('provider_attempted',True) for r in model.requests),
            'wrapper_attempts':len(model.requests),'results_file':'results.json',
            'human_review_status':'pending','automatic_checks_are_not_semantic_quality_scores':True}
    (output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['seal','run'])
    parser.add_argument('--cases',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-provider-requests',type=int,choices=range(1,37),default=36)
    parser.add_argument('--origin',choices=['developer','external'],default='developer')
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    if args.action=='seal':
        if not args.cases:parser.error('seal requires --cases')
        result=seal(args.cases,args.output,args.max_provider_requests,args.origin)
    else:
        if not args.live:parser.error('run requires --live')
        result=run(args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in {'cases'}},ensure_ascii=False))


if __name__=='__main__':main()

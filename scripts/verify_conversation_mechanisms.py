"""Bounded real extraction and two-turn verification; never retry a saved phase."""
import argparse
import json
from pathlib import Path

from app.chat import SelectionAssistant
from app.chat_intent import understand_intent
from app.chat_schemas import ChatMessage, ChatRequest, SelectionProfile
from app.chat_store import ChatStore
from app.conversation_policy import route_turn
from app.selection_catalog import load_catalog
from scripts.verify_game_followup import BoundedModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', required=True)
    parser.add_argument('--phase', choices=['intent', 'conversation', 'reference'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    marker = args.output / (args.phase + '.started')
    marker.open('x').close()
    model = BoundedModel(args.output, limit=18)
    requests = args.output / 'requests.json'
    if requests.exists():
        model.requests = json.loads(requests.read_text(encoding='utf-8'))
    if not model.available:
        raise ValueError('No configured model')
    if args.phase == 'intent':
        cases = [
            ('portrait', ['拍照','续航'], '你更常拍什么题材？', '主要拍孩子和家人的人像', {}, 'refine'),
            ('duration', ['游戏','续航'], '每次一般连续玩多久？', '一般一玩就是两小时', {}, 'refine'),
            ('threshold', ['拍照','续航'], '还有什么要求？', '机身必须不超过180克', {}, 'revise'),
            ('mixed', ['游戏','续航'], '还有什么要求？', '也想兼顾拍照，常拍人像', {}, 'revise'),
            ('cancel', ['游戏','续航'], '还有什么要求？', '不要求高刷了，其他照旧', {'high_refresh_required':True}, 'revise'),
            ('reference', ['游戏','续航'], '你更想了解哪款？', '第二款的屏幕刷新率是多少？', {}, 'question'),
            ('alternatives', ['游戏','续航'], '想继续了解吗？', '这几个都不太合适，再找些别的吧', {}, 'alternatives'),
            ('budget', ['游戏','续航'], '想继续了解吗？', '预算改成四千五以内，其他不变', {}, 'revise'),
        ]
        catalog = load_catalog()
        ids = list(catalog)[:3]
        rows = []
        for name, topics, followup, text, extra, expected in cases:
            profile = SelectionProfile(budget_max=6000, priorities=topics,
                candidate_ids=ids, excluded_brands=['华为'], **extra)
            history = [ChatMessage(role='user',content='预算6000，'+ '和'.join(topics)+'，不要华为'),
                ChatMessage(role='assistant',content='先看这几款。', followup=followup,
                    cards=[catalog[pid].card(profile) for pid in ids])]
            before = profile.model_dump()
            intent = understand_intent(profile, text, catalog, model, history)
            route = route_turn(before, profile, intent, has_candidates=True, selected=[])
            rows.append(dict(name=name,input=text,before=before,after=profile.model_dump(),
                intent=vars(intent),route=vars(route),expected_route=expected,
                route_matches_expectation=route.operation==expected))
            (args.output/'intent-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'phase':'intent','calls':len(model.requests),'routes':[r['route']['operation'] for r in rows]}))
    elif args.phase == 'reference':
        summary=json.loads((args.output/'summary.json').read_text(encoding='utf-8'))
        assistant=SelectionAssistant(ChatStore(Path('runs/chat-sessions.sqlite3')),model_client=model)
        assistant.agent_policy=assistant.agent_policy.model_copy(update={'max_model_calls':model.limit-len(model.requests)})
        result=assistant.turn(ChatRequest(message='第二款的屏幕刷新率是多少？',session_id=summary['session_id'],search_mode='local'))
        (args.output/'reference.json').write_text(result.model_dump_json(indent=2),encoding='utf-8')
        print(json.dumps({'phase':'reference','calls':len(model.requests),'session_id':result.session_id}))
    else:
        # Separate new conversation in the normal store, available in the current UI.
        assistant = SelectionAssistant(ChatStore(Path('runs/chat-sessions.sqlite3')), model_client=model)
        sid = None
        rows = []
        for index, text in enumerate(['预算6000，主要玩游戏，注重续航','主要玩绝区零'],1):
            if len(model.requests) >= model.limit:
                rows.append({'input':text,'status':'skipped_request_limit'})
                break
            # Keep at least four requests for the second turn, with the same
            # production limits otherwise. The global 18-request cap is shared
            # with extraction diagnostics and never renewed between phases.
            assistant.agent_policy = assistant.agent_policy.model_copy(update={
                'max_model_calls':min(6, model.limit-len(model.requests)-(4 if index==1 else 0))})
            session = assistant.turn(ChatRequest(message=text,session_id=sid,search_mode='web'))
            sid = session.session_id
            (args.output/f'turn-{index}.json').write_text(session.model_dump_json(indent=2),encoding='utf-8')
            rows.append({'input':text,'reply':session.messages[-1].model_dump(),'profile':session.profile.model_dump()})
        summary = dict(session_id=sid,max_provider_requests=18,actual_provider_requests=len(model.requests),
            turns=rows,human_review_status='pending')
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps({'session_id':sid,'calls':len(model.requests)}))


if __name__ == '__main__':
    main()

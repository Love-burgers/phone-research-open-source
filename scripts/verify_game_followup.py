"""Explicit, bounded two-turn verification; retains failures and request usage."""
import argparse
import json
import os
from pathlib import Path
from time import monotonic

from app.chat import SelectionAssistant
from app.chat_schemas import ChatRequest
from app.chat_store import ChatStore
from app.model_api import ModelClient
from app.tool_loop import ExecutionLimit


class BoundedModel(ModelClient):
    def __init__(self, output, limit=8):
        super().__init__()
        self.output,self.limit,self.requests=output,limit,[]

    def call(self,kind,messages,run,**extra):
        if len(self.requests)>=self.limit:
            raise ExecutionLimit('verification_request_limit')
        event={'number':len(self.requests)+1,'kind':kind,'input':[
            m if isinstance(m,dict) else m.model_dump(mode='json') for m in messages],**extra}
        self.requests.append(event)
        self.last_usage=None
        started=monotonic()
        try:
            result=run()
            event['output']=result if isinstance(result,dict) else {'content':result.content,'tool_calls':result.tool_calls}
            return result
        except Exception as error:
            event['error_type']=type(error).__name__
            event['provider_attempted']=not isinstance(error,ExecutionLimit)
            raise
        finally:
            event.setdefault('provider_attempted',True)
            event.update(usage=self.last_usage,elapsed_seconds=round(monotonic()-started,4))
            (self.output/'requests.json').write_text(json.dumps(self.requests,ensure_ascii=False,indent=2),encoding='utf-8')

    def invoke_tools(self,messages,tools,**kwargs):
        return self.call('tools',messages,lambda:super(BoundedModel,self).invoke_tools(messages,tools,**kwargs),tools=tools)

    def complete_json(self,messages,**kwargs):
        return self.call('json',messages,lambda:super(BoundedModel,self).complete_json(messages,**kwargs))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ui',action='store_true',help='Create a new verification conversation in the normal UI store; never rewrite an old one.')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    model=BoundedModel(args.output)
    if not model.available:raise ValueError('No configured model')
    path=Path(os.getenv('CHAT_STORAGE_PATH') or 'runs/chat-sessions.sqlite3') if args.ui else args.output/'sessions.sqlite3'
    service=SelectionAssistant(ChatStore(path),model_client=model)
    sid=None;rows=[]
    for number,text in enumerate(['预算6000，主要玩游戏，注重续航','主要玩原神'],1):
        if len(model.requests)>=model.limit:
            rows.append({'input':text,'status':'skipped_verification_budget'});break
        result=service.turn(ChatRequest(message=text,session_id=sid,search_mode='web'))
        sid=result.session_id
        (args.output/f'turn-{number}.json').write_text(result.model_dump_json(indent=2),encoding='utf-8')
        reply=result.messages[-1]
        rows.append({'input':text,'content':reply.content,'blocks':[b.text for b in reply.blocks],
            'cards':[{'name':c.name,'price':c.price} for c in reply.cards],
            'metrics':reply.agent_trace.get('metrics') if reply.agent_trace else None,
            'stop_reason':reply.agent_trace.get('stop_reason') if reply.agent_trace else None})
    summary={'session_id':sid,'search_mode':'web','max_provider_requests':model.limit,'actual_provider_requests':len(model.requests),
        'turns':rows,'human_review_status':'pending','scope':'One two-turn live run; no retries or prefilled quality scores.'}
    (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'session_id':sid,'provider_requests':len(model.requests),'turns':len(rows)}))


if __name__=='__main__':main()

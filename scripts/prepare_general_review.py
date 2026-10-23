"""Package actual runs and older visible comparisons without grading them."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .prepare_chat_review import review_turn


def prepare(root, previous, output, suffix='navigation', mixed_suffix='delivery'):
    cases, provenance = [], []
    for directory, title in (
        (f'user-001-{suffix}', '本轮：7500元、一款、游戏与高刷'),
        (f'user-002-{suffix}', '本轮：7500 → 两款游戏 → 7000左右'),
        (f'mixed-photo-{mixed_suffix}', '新增场景：拍孩子、续航、重量 → 取消条件 → 夜景追问'),
    ):
        path = root / directory / 'result.json'
        record = json.loads(path.read_text(encoding='utf-8'))
        if not record.get('completed_at') or record.get('failure'):
            raise ValueError(f'Run not complete: {directory}')
        if [s['messages'][-2]['content'] for s in record['turns']] != record['original_messages']:
            raise ValueError('Recorded inputs differ')
        turns, before = [], None
        for i, session in enumerate(record['turns'], 1):
            turns.append(review_turn(session, f'{directory}-{i}', before))
            before = session['profile']
        cases.append({'id': directory, 'title': title, 'turns': turns,
            'provenance': '实际模型和公开工具运行，原答、失败与来源保留；尚无人评。',
            'reported_feedback': ['用户本次要求：只更新本轮明确变化的条件。',
                '用户本次要求：不要强行凑满推荐数量，也不要为了产生变化而无理由更换候选。',
                '用户本次要求：不要把没有资料写成“不满足”，也不要把提到某个关键词当成已证明满足。']})
        provenance.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    old = json.loads((previous / 'blind-packet.json').read_text(encoding='utf-8'))
    for case in deepcopy(old['cases']):
        case['id'] = 'history-' + case['id']
        case['title'] = '上一轮留存对照：' + case['title']
        for turn in case['turns']:
            turn['id'] = 'history-' + turn['id']
        cases.append(case)
    output.mkdir(parents=True, exist_ok=False)
    packet = {'schema_version': 2, 'review_kind': 'home_chat',
        'created_at': datetime.now(timezone.utc).isoformat(), 'cases': cases,
        'instructions': '可只评看过的一轮，或只写意见，不必全部评。修改前后均可见，属于产品对照复核，不是独立盲审；未代填人工判断。中间失败原件也保留在运行目录。'}
    packet['packet_id'] = hashlib.sha256(json.dumps(cases, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:24]
    (output/'blind-packet.json').write_text(json.dumps(packet,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'preparation.json').write_text(json.dumps({'records':provenance,'previous_packet':str(previous),
        'human_review_status':'pending','new_model_calls':0,'turns':sum(len(c['turns']) for c in cases)},
        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return {'cases':len(cases),'turns':sum(len(c['turns']) for c in cases),'human_review_status':'pending'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous',type=Path,default=Path('runs/user-feedback-review-20261022-v2'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--suffix',default='navigation')
    parser.add_argument('--mixed-suffix',default='delivery')
    args=parser.parse_args()
    print(json.dumps(prepare(args.root,args.previous,args.output,args.suffix,args.mixed_suffix)))


if __name__=='__main__':
    main()

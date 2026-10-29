"""Bounded dialogue memory shared by intent, planning and answer requests."""

import re

from .evidence_rules import game_in_text
from .config import get_settings
from .feature_requirements import REPORT_FIELDS


def shorten(text, limit):
    if len(text) <= limit:
        return text
    return text[:limit*2//3]+' … '+text[-limit//3:]


def evidence_excerpt(quote, limit):
    if len(quote) <= limit:
        return quote
    if '资料边界：' in quote:
        boundary = quote[quote.index('资料边界：'):]
        boundary = shorten(boundary, min(len(boundary), limit//2))
        return quote[:limit-len(boundary)-3]+' … '+boundary
    return shorten(quote, limit)


def recent_dialogue(messages, *, limit=None, characters=None):
    """Keep user turns, followups and every displayed candidate, not a first-phone prefix."""
    settings = get_settings()
    limit = settings.chat_history_messages if limit is None else limit
    characters = settings.chat_history_characters if characters is None else characters
    selected = list(messages[-limit:])
    if not selected:
        return []
    per_message = max(100, characters//len(selected))
    rows = []
    for message in selected:
        if message.role == 'user':
            rows.append({'role': message.role, 'content': shorten(message.content, per_message)})
            continue
        parts = [shorten(message.content, per_message//3)] if message.content else []
        if message.followup:
            parts.append(shorten(message.followup, per_message//3))
        if message.blocks:
            size = max(30, per_message//(len(message.blocks)+2))
            parts.extend(block.phone_id+'：'+shorten(block.text, size) for block in message.blocks)
        row = {'role': message.role, 'content': '\n'.join(parts)}
        if message.cards:
            row['candidate_order'] = [{'id': c.id, 'name': c.name} for c in message.cards]
        rows.append(row)
    return rows


def focused_games(profile, turn):
    if turn.get('task') is not None:return list(turn['task'].get('games',[]))
    focus = turn.get('response_focus') or turn.get('message', '')
    terms = turn.get('focus_terms') or []
    current = [g for g in profile.games if g in terms or game_in_text(g, focus)]
    return current or list(profile.games)


def query_memory(messages, names, *, limit=4):
    records, seen = [], set()
    for message in reversed(messages):
        if message.role != 'assistant':
            continue
        for action in reversed((message.agent_trace or {}).get('actions', [])):
            if action.get('action') != 'tool' or action.get('name') not in {'web_search', 'fetch_page'}:
                continue
            args = action.get('arguments', {})
            if args.get('phone_name') not in names:
                continue
            key = (action['name'], str(args))
            if key in seen:
                continue
            seen.add(key)
            result = action.get('result', {})
            records.append({'tool': action['name'], 'arguments': args, 'status': result.get('status'),
                            'search_feedback': result.get('search_feedback'), 'fetch_feedback': result.get('fetch_feedback')})
            if len(records) >= limit:
                return list(reversed(records))
    return list(reversed(records))


def asks_price(text):
    for clause in re.split(r'[，,。；;]|但是|不过|但', text):
        if not re.search(r'预算|价格|多少钱|价位|左右|推荐|选购|买|\d+\s*元', clause):
            continue
        if re.search(r'(?:不用|不必|不要|无需|不问|不看|不谈|别|勿)[^，。；]{0,12}(?:报|提|说|问|看|查)?(?:价格|报价|售价|预算|价位)', clause):
            continue
        return True
    return False


def answer_requirements(existing,text):
    """Evidence presentation preferences, never product qualification gates."""
    values=list(existing)
    for clause in re.split(r'[，,。；;\n]',text):
        for label,pattern in REPORT_FIELDS.items():
            if not re.search(pattern,clause):continue
            if re.search(r'取消|不用|无需|不必|不需要',clause):values=[v for v in values if v!=label]
            elif label not in values:values.append(label)
    return values[-8:]

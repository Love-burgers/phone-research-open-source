"""One bounded evidence merge policy for tools, saved sessions and fallback RAG."""
import re

from .phone_identity import source_identity_matches
from .source_retention import retained_sources
from .selection_requirements import local_only, recommendation_gaps
from .feature_requirements import feature_keys, feature_values, field_keys, observed_fields
from .selection_prices import source_price_options, sanitize_price_source
from .evidence_rules import measured_game_parts, game_measurements, owned_fact_text
from .selection_presentation import configuration_facts


def price_question(turn):
    if turn.get('task') is not None:return turn['task']['kind']=='price'
    return not turn.get('recommendation_turn') and bool(re.search(
        r'价格|售价|价钱|多少钱|报价|到手价|卖多少|现价',turn['message']))


def refresh_kinds(message):
    if not re.search(r'刷新|重新查|更新资料|最新资料|最新价格|今天.*价',message):return set()
    kinds=set()
    if re.search(r'价格|售价|报价|价钱|多少钱|到手价',message):kinds.add('price')
    if re.search(r'配置|参数|规格|NFC|无线充电|屏幕',message,re.I):kinds.add('spec')
    if re.search(r'评测|实测|测试',message):kinds.add('review')
    return kinds or {'spec','price','review'}


def public_allowed(turn):
    if turn.get('task') is not None:return turn['task']['public_allowed']
    return turn.get('search_mode')!='local' and not local_only(turn['message'])


def merge_sources(state,phone_id,incoming,catalog,profile,*,focus_topics=(),refresh=(),fresh=False):
    returned=[];added=0
    for source in incoming:
        source=source.model_copy(update={'quote':owned_fact_text(source.quote)})
        source=sanitize_price_source(source,catalog[phone_id].name)
        if source.phone_id!=phone_id or not source_identity_matches(catalog[phone_id].name,source.title,source.quote):continue
        if source.evidence_kind in refresh:
            status='verified' if fresh and source.source_type=='page' else 'pending'
            source=source.model_copy(update={'refresh_status':status})
        old=state['sources'].get(source.id)
        if old is not None and (old.phone_id!=phone_id or old.url!=source.url):continue
        if old is None:
            old=next((s for s in state['sources'].values() if s.phone_id==phone_id and s.url==source.url
                and (s.source_type=='snippet' and source.source_type=='page'
                     or s.source_type==source.source_type=='page' and s.evidence_kind==source.evidence_kind
                        and s.price_variant==source.price_variant
                     or s.quote==source.quote and s.evidence_kind==source.evidence_kind)),None)
        if old:
            same_page=(source.source_type=='page' and old.source_type in {'page','knowledge'}
                and source.checked_on>=old.checked_on
                and (source.evidence_kind==old.evidence_kind or old.evidence_kind in {'spec','unknown'}))
            if old.source_type=='snippet' and source.source_type=='page' or same_page:
                source=source.model_copy(update={'id':old.id})
                added+=source!=old
                state['sources'][old.id]=source
            else:source=old
        else:
            existing=[s for s in state['sources'].values() if s.phone_id==phone_id]
            if len(existing)>=8:
                kept=retained_sources([*existing,source],catalog[phone_id],profile,focus_topics=focus_topics)
                if source not in kept:continue
                evicted=[s for s in existing if s not in kept]
                for victim in evicted:del state['sources'][victim.id]
                state.setdefault('actions',[]).append({'action':'source_retention','phone_id':phone_id,
                    'admitted_id':source.id,'evicted_ids':[s.id for s in evicted],
                    'reason':'bounded_marginal_coverage_and_recency'})
            state['sources'][source.id]=source;added+=1
        state['allowed_urls'].add((phone_id,source.url));returned.append(source.model_dump())
    return [s for s in returned if s['id'] in state['sources']],added


def session_evidence(turn,catalog,scope):
    state={'sources':{},'allowed_urls':set(),'actions':[]}
    saved={};refresh=refresh_kinds(turn['message'])
    for message in reversed(turn['session'].messages):
        if message.role!='assistant':continue
        for source in message.sources:
            if source.phone_id not in scope:continue
            source=source.model_copy(deep=True)
            if source.evidence_kind in refresh:
                source.refresh_status='pending' if public_allowed(turn) else 'not_permitted'
            saved.setdefault(source.id,source)
    # Keep the latest snapshot for each ID; opposing URLs remain separate.
    for source in reversed(list(saved.values())):
        merge_sources(state,source.phone_id,[source],catalog,turn['session'].profile,
                      focus_topics=turn.get('focus_topics',()))
    return state


def finish_refresh(turn,state,*,attempted):
    if not refresh_kinds(turn['message']):return
    verified=[s for s in state['sources'].values() if s.refresh_status=='verified']
    for sid,source in list(state['sources'].items()):
        if source.refresh_status=='pending':
            used=attempted if isinstance(attempted,bool) else source.phone_id in attempted
            status='failed' if used else 'not_attempted' if public_allowed(turn) else 'not_permitted'
            for newer in verified:
                if (source.phone_id,source.url,source.evidence_kind,source.market)!=(newer.phone_id,newer.url,newer.evidence_kind,newer.market) or newer.checked_on<source.checked_on:continue
                old_versions={o.variant for o in source_price_options(source)}
                new_versions={o.variant for o in source_price_options(newer)}
                if source.evidence_kind!='price' or old_versions and old_versions<=new_versions:
                    status='superseded';break
            state['sources'][sid]=source.model_copy(update={'refresh_status':status})


def refresh_notices(sources,turn=None):
    notices=[]
    current=refresh_kinds(turn['message']) if turn else None
    relevant=(current or ({'price'} if price_question(turn) else {'review','spec'})) if turn else None
    successful={(s.phone_id,s.evidence_kind) for s in sources if s.refresh_status=='verified'}
    for source in sources:
        if relevant is not None and source.evidence_kind not in relevant:continue
        if source.refresh_status in {'failed','not_permitted','not_attempted','pending'}:
            reason={'not_permitted':'本轮未允许公开刷新','not_attempted':'本轮刷新未执行',
                    'failed':'本轮刷新未取得可核验的新记录','pending':'本轮刷新尚未确认'}[source.refresh_status]
            if (source.phone_id,source.evidence_kind) in successful:
                reason='本轮已核实部分新记录；其余来源或版本未更新'
            if current is not None and source.evidence_kind not in current:
                reason=reason.replace('本轮','此前')
            notices.append(f'{reason}；保留{source.checked_on}核对的历史'+('价格，不能当作最新价。' if source.evidence_kind=='price' else '资料，仍按原日期与条件使用。'))
    return list(dict.fromkeys(notices))


def evidence_gaps(turn,phone,sources):
    body=[s for s in sources if s.phone_id==phone.id and s.source_type!='snippet']
    if turn.get('recommendation_turn'):return recommendation_gaps(phone,turn['session'].profile,body)
    gaps=[]; keys=set(turn.get('task',{}).get('fields',field_keys(turn['message'])))
    observed={key for s in body for key in observed_fields(s.quote)}
    gaps.extend(key+'未取得属性正文' for key in keys-observed)
    if price_question(turn) and not any(source_price_options(s) and s.refresh_status not in {'pending','failed','not_attempted','not_permitted'} for s in body):
        gaps.append('当前版本价待核对')
    games=turn.get('task',{}).get('games',[])
    if (games or re.search(r'游戏|原神|王者|星穹铁道|帧率|发热|温度',turn['message'])) and not price_question(turn):
        requested={label for label,pattern in {'帧率':r'帧率|多少帧|fps','温度':r'发热|温度|温升','功耗':r'功耗','耗电':r'耗电|掉电|电量'}.items()
                   if re.search(pattern,turn['message'],re.I)}
        for game in games or [None]:
            reviews=[s for s in body if s.evidence_kind=='review']
            covered=set().union(*(game_measurements(s.quote,game) for s in reviews))
            if requested:
                gaps.extend((game or '游戏')+label+'缺少对应条件的实测正文' for label in sorted(requested-covered))
            elif not configuration_facts(phone,['游戏'],body) and not any(any(not game or game in part for part in measured_game_parts(s.quote)) for s in reviews):
                gaps.append((game or '当前问题')+'缺少对应测试正文')
    if not body:gaps.append('没有可引用正文')
    return gaps


def missing_reply(turn):
    if price_question(turn):return '这轮没核实到所问机型的最新版本价格；已取得的历史记录仍按原日期保留，不能作为最新到手价。'
    if turn.get('alternative_ids'):return '暂时没核实到其他合适机型；本次查询不足以确认新的备选，不代表全网没有资料。'
    if turn.get('recommendation_turn'):return '这轮没核实到合适机型，查到的线索还不足以推荐；已有条件会保留。'
    focus=turn.get('response_focus') or turn['message']
    return '关于“'+focus[:120]+'”，这轮没有取得可核验的对应正文；现有资料不足以判断该项表现，不代表全网没有相关资料。'

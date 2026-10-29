"""Textual evidence checks; parameters alone never establish measured outcomes."""

import re
from decimal import Decimal
from .topics import COMPUTE_PATTERNS
from .phone_identity import owner_at, model_spans


PARAMETER = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*(mAh|W|Hz|赫兹|GB|TB|mm|毫米|g|克|英寸|万(?:像素)?|亿(?:像素)?|MP)(?![a-z])", re.I)

PARAMETER_ROLES = {
    'hz': [('touch_sampling',r'触控|采样'),('display_refresh',r'屏幕|刷新率|高刷|显示|ProMotion')],
    'mm': [('thickness',r'厚度|薄至'),('width',r'宽度|宽为'),('height',r'高度|长度')],
    'w': [('wireless_charge',r'无线'),('wired_charge',r'有线')],
    'pixels': [('front_camera',r'前置'),('main_camera',r'主摄'),('telephoto',r'长焦'),('ultrawide',r'超广角')],
}


FOREIGN_FACT = r'行业(?:最高|最低|平均)|同类(?:最高|最低|平均)|竞品(?:参数|配置|推荐)?|相关推荐|广告推荐'


def owned_fact_text(text):
    """Project old cached excerpts onto labelled product facts, without changing history.

    A comparison label owns the remainder of its segment; a slash/newline starts
    a new product field. Numeric magnitude never determines ownership.
    """
    return ' / '.join(part for segment in re.split(r' / |\n',text)
        if (part:=re.split(FOREIGN_FACT,segment,maxsplit=1)[0].strip(' ：:，,；;')))


def parameter_values(text):
    """Bind numbers to nearby labels within a clause, preserving dimension roles."""
    for match in PARAMETER.finditer(text):
        number, unit = Decimal(match[1]), match[2].lower()
        unit = {'毫米':'mm','克':'g','赫兹':'hz'}.get(unit,unit)
        if unit == 'tb':
            number, unit = number*1024, 'gb'
        if unit.startswith(('万','亿')) or unit=='mp':
            if unit!='mp' and not unit.endswith('像素') and not re.search(
                    r'摄像|相机|像素|主摄|长焦|超广角|前置',text[max(0,match.start()-20):match.end()+20]):
                continue
            number *= 10000 if unit.startswith('万') else 100000000 if unit.startswith('亿') else 1000000
            unit = 'pixels'
        before = re.split(r'[，,。；;|/]',text[:match.start()])[-1][-20:]
        if unit=='hz':before=re.split(r'[。；;|/]',text[:match.start()])[-1][-64:]
        after = re.split(r'[，,。；;|/]',text[match.end():])[0][:16]
        if unit=='w':
            before=re.split(r'[，,。；;|/、]',text[:match.start()])[-1][-20:]
            after=re.split(r'[，,。；;|/、]',text[match.end():])[0][:16]
            next_value=PARAMETER.search(after)
            if next_value:after=after[:next_value.start()]
        if unit=='pixels':
            # Camera lists consistently use either value→role or role→value.
            # A role following the previous number must not label the next one.
            prefix=re.split(r'[，,。；;|/\n]',text[:match.start()])[-1]
            suffix=re.split(r'[，,。；;|/\n]',text[match.end():])[0]
            clause=prefix+match[0]+suffix
            first_quantity=next((m for m in PARAMETER.finditer(clause)
                                 if m[2].lower().startswith(('万','亿','mp'))),None)
            first_role=re.search(r'前置|主摄|长焦|超广角',clause)
            if first_quantity and first_role:
                # Introductory prose may mention several lenses before the
                # actual list. Only a bare label adjacent to the first value
                # establishes role→value order; prose is not a column label.
                label_first=(first_role.end()<=first_quantity.start() and re.fullmatch(
                    r'(?:摄像头|镜头)?(?:像素|分辨率)?\s*(?:为|是|采用|配备|搭载)?\s*[：:]?\s*',
                    clause[first_role.end():first_quantity.start()]))
                if label_first:after=''
                else:before=''
        candidates=[]
        for role, pattern in PARAMETER_ROLES.get(unit,[]):
            candidates += [(len(before)-m.end()+.5,role) for m in re.finditer(pattern,before,re.I)]
            candidates += [(m.start(),role) for m in re.finditer(pattern,after,re.I)]
        role=min(candidates)[1] if candidates else None
        yield number,unit,role,match


def parameter_observations(sources):
    facts=[(v,u,r,s) for s in sources if s.source_type!='snippet' for v,u,r,m in parameter_values(owned_fact_text(s.quote))]
    newest={}
    for v,u,r,s in facts:
        key=(s.phone_id,s.url,u,r)
        newest[key]=max(newest.get(key,''),s.checked_on)
    return [(v,u,r,s) for v,u,r,s in facts if s.checked_on==newest[(s.phone_id,s.url,u,r)]]


def unsupported_parameters(text, sources, requirements=()):
    """Check scalar hardware claims against same-phone cited bodies.

    This checks literal values/units, not whether a device is good at a task.
    Price, benchmark conditions and qualitative claims have separate guards.
    """
    # A rejected user threshold is a condition, not a claimed hardware value.
    # Only exempt its literal appearance in an explicitly negative comparison.
    for requirement in requirements:
        start = text.find(requirement)
        if start >= 0 and re.search(r'不满足|不符合|未达到|低于', text[max(0, start-12):start]):
            text = text[:start] + text[start:].replace(requirement, '', 1)
    verified = {(value,unit,role) for value,unit,role,source in parameter_observations(sources)}
    issues = [match[0] for value,unit,role,match in parameter_values(text)
              if not any(value==v and unit==u and (role is None or role==r) for v,u,r in verified)]
    for role,patterns in COMPUTE_PATTERNS.items():
        verified_counts={m['count'] for source in sources if source.source_type!='snippet'
                         for pattern in patterns for m in re.finditer(pattern,owned_fact_text(source.quote),re.I)}
        issues.extend(m[0] for pattern in patterns for m in re.finditer(pattern,text,re.I) if m['count'] not in verified_counts)
    # Chip identifiers are identities, not arbitrary occurrences of a model
    # code in a support article. Require chip-labelled source context.
    for match in re.finditer(r'(?<![A-Za-z0-9])A\d{2}(?:\s*Pro)?(?![A-Za-z0-9])',text,re.I):
        if not any(re.search(re.escape(match[0]),owned_fact_text(s.quote),re.I) and any(
                re.search(r'芯片|仿生|处理器|CPU|SoC|搭载',owned_fact_text(s.quote)[max(0,m.start()-24):m.end()+24],re.I)
                for m in re.finditer(re.escape(match[0]),owned_fact_text(s.quote),re.I))
                for s in sources if s.source_type!='snippet'):
            issues.append(match[0])
    # Named hardware capabilities need literal support too. Pixel counts alone
    # do not establish a periscope design, macro mode or stabilization type.
    bodies=[owned_fact_text(source.quote) for source in sources if source.source_type!='snippet']
    def capabilities(value,pattern):
        for match in re.finditer(pattern,value,re.I):
            prefix=value[max(0,match.start()-8):match.start()]
            suffix=value[match.end():match.end()+12]
            negative=bool(re.search(r'(?:不支持|不具备|未配备|没有|不带|无)\s*$',prefix))
            if negative and re.match(r'[^，。；,;]{0,6}(?:实测|测试|评测|记录|数据)',suffix):
                continue  # Missing test evidence is not a missing feature.
            yield negative,match[0]
    for pattern in (r'潜望(?:式)?',r'微距',r'浮动镜组',r'(?<![A-Za-z])OIS(?![A-Za-z])|光学防抖',r'(?<![A-Za-z])EIS(?![A-Za-z])|电子防抖'):
        supported={negative for quote in bodies for negative,_ in capabilities(quote,pattern)}
        issues.extend(label for negative,label in capabilities(text,pattern) if negative not in supported)
    return issues

def photo_observation(text):
    aspect = r"曝光|白平衡|细节|肤色|色彩|色调|对焦|解析力|高光|暗部|噪点|虚化"
    outcome = r"清晰|准确|自然|鲜活|稳定|保留|丢失|压制|过曝|偏暖|偏冷|模糊|涂抹|跑偏|不准|没有差别|几乎没有差别"
    return bool(re.search(r"(?:" + aspect + r").{0,16}(?:" + outcome + r")|(?:" + outcome + r").{0,10}(?:" + aspect + r")", text))


def comparison_scope_mismatch(text, count):
    """A comparison's stated group must agree with the displayed answer group."""
    numbers = dict(zip('一二三四五六七八九十', range(1, 11))) | {'两': 2}
    if any((int(m[1]) if m[1].isdigit() else numbers.get(m[1])) >= count
        for m in re.finditer(r'前([一二两三四五六七八九十]|\d+)\s*(?:款|台|部)',text)):return True
    return any((int(m[1]) if m[1].isdigit() else numbers.get(m[1])) != count
               for m in re.finditer(r'([一二两三四五六七八九十]|\d+)\s*(?:款|台|部)(?:之中|中|里)', text))


def hardware_extrapolation(text):
    """Reject specific display→render/FPS guarantees, not qualitative reasoning."""
    display_load = re.search(r'(?:[124]K\s*屏|屏幕分辨率|\d{3,4}[×xX*]\d{3,4}屏)[^。；]{0,50}(?:高负载|负载高|增加负载|更耗电|降低帧率)|高分辨率负载', text, re.I)
    if re.search(r'不代表|不能(?:据此|证明|推断)|无法推断|不等于|未必|若.*(?:原生|分辨率)|如果',text):
        display_load = None
    fps_bridge=re.search(r'(?:因此|所以|意味着|说明)[^。；]{0,55}(?:帧率上限|游戏帧率|能跑|能达到)[^。；]{0,35}(?:Hz|不受|不再受|无限制)|(?:帧率上限|游戏帧率)[^。；]{0,45}Hz[^。；]{0,12}(?:限制|解锁)',text,re.I)
    if re.search(r'不代表|不能据此|不能推断|不等于',text):fps_bridge=None
    return bool(display_load or fps_bridge)

def endurance_observation(text):
    return bool(re.search(r"测试|实测|实际|连续|播放|模拟", text)
                and re.search(r"剩余|剩下|耗电|电量|测得", text)
                and re.search(r"\d+(?:\.\d+)?\s*(?:小时|分钟|%|％)", text))

def game_in_text(game, text):
    normalize = lambda s: re.sub(r"[^\w]", "", s).casefold()
    wanted, content = normalize(game), normalize(text)
    return wanted in content or "星穹铁道" in wanted and bool(re.search(r"星穹铁道|星铁", content))

def canonical_games(games):
    return list(dict.fromkeys("崩坏：星穹铁道" if re.search(r"星穹铁道|星铁", game) else game for game in games))[:8]


def measurement_records(text,phone_name,games=(),*,claim=False,spec=False):
    """Literal FPS/temperature/endurance claims with subject and measurement role."""
    names=list(dict.fromkeys([*games,'原神','王者荣耀','和平精英','崩坏：星穹铁道','星穹铁道','星铁','绝区零',*re.findall(r'《([^》]+)》',text)]))
    markers=sorted((m.start(),m.end(),g) for g in names for m in re.finditer(re.escape(g),text))
    pattern=(r'(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>fps|帧|℃|°C|摄氏度|度|小时|分钟)'
             r'|(?P<fps_label>(?:平均)?帧率\s*(?:约|为|达到|稳定在|上限(?:为)?|[:：])?\s*)(?P<fps_value>\d+(?:\.\d+)?)(?:\s*(?:fps|帧))?')
    for m in re.finditer(pattern,text,re.I):
        prefix=re.split(r'[，,。；;\n/、]',text[:m.start()])[-1][-45:]
        suffix=re.split(r'[，,。；;\n/、]',text[m.end():])[0][:18]
        if claim and (re.search(r'能否|是否|尚未核实(?:能否)?|不等于|不能保证',prefix[-16:]) or re.match(r'\s*(?:未知|待核实|尚未确认|仍待验证)',suffix)):
            continue
        unit=(m['unit'] or 'fps').lower()
        if unit in {'fps','帧'}:
            measured=bool(m['fps_label'] or re.search(r'平均|测得|实测|达到|稳定在|帧率为',prefix[-16:]))
            role='fps_setting' if not measured and (re.search(r'设定|设置|锁定|目标|画质',prefix[-16:]) or re.match(r'\s*(?:设置|设定|档|选项|模式)',suffix)) else 'fps_result'
            if re.search(r'视频录制|录制视频|录像|视频拍摄|视频格式|帧录制',prefix+m[0]+suffix):role='declared_video_recording'
        elif unit in {'℃','°c','摄氏度','度'}:
            if unit=='度' and re.search(r'广角|视角|角度|旋转',prefix+suffix):continue
            role='room_temperature' if re.search(r'室温|环境温度',prefix) else 'device_temperature_rise' if '温升' in prefix else 'device_temperature'
        else:
            labels=re.findall(r'视频播放|音频播放|续航|充电|充入|充至|充满',prefix)
            if labels and labels[-1] in {'充电','充入','充至','充满'} or not labels and re.search(r'充电|充入|充至|充满',suffix):continue
            if not re.search(r'续航|视频播放|音频播放|(?:能|可|最长).{0,10}(?:使用|玩|撑|播放)|(?:使用|玩|撑|播放).{0,8}(?:可达|最长)',prefix):continue
            media_labels=re.findall(r'视频|音频|游戏|原神|王者|星铁|铁道',prefix)
            media={'视频':'video','音频':'audio'}.get(media_labels[-1],'game') if media_labels else 'use'
            role=('declared_' if spec or re.search(r'标称',prefix) else 'measured_')+media+'_endurance'
        before=[g for g in markers if g[0]<m.start()]
        active=before[-1][2] if before else games[0] if claim and len(games)==1 else None
        active='崩坏：星穹铁道' if active and re.search(r'星穹铁道|星铁',active) else active
        start=before[-1][0] if before else 0
        following=[g[0] for g in markers if g[0]>m.end()]
        end=min(following) if following else len(text)
        owner=owner_at(text,m.start()+1,phone_name)
        if claim and re.search(r'本机|这款手机',prefix):owner=owner_at('',0,phone_name)
        conditions=text[start:end]
        if claim:conditions=re.split(r'但是|但|不代表|不能直接|不能外推|不等于|尚未|仍需',conditions,maxsplit=1)[0]
        modes={mode[0] for mode in re.finditer(r'插帧|超帧|补帧|原生',conditions)
               if not re.search(r'关闭|未开启|不启用|非|未',conditions[max(0,mode.start()-4):mode.start()])}
        quality=set(re.findall(r'(?:最高|极高|极低|高|中|低)画质',conditions))
        yield {'role':role,'value':Decimal(m['value'] or m['fps_value'])*(60 if unit=='小时' else 1),'owner':owner,'game':active,
            'conditions':conditions,'durations':_measured_durations(conditions),'modes':modes,'quality':quality,'match':m}


def unsupported_measurements(text,sources,phone_name,games=()):
    verified=[]
    wanted=owner_at('',0,phone_name)
    for source in sources:
        if source.source_type=='snippet':continue
        quote=owned_fact_text(source.quote)
        passages=measured_game_parts(quote) if source.evidence_kind=='review' else []
        if source.evidence_kind=='spec':passages.append(quote)
        if source.evidence_kind=='review' and re.search(r'续航.{0,8}(?:实测|测试)|实测.{0,8}续航|测得|耗尽',quote):passages.append(quote)
        for part in passages:
            for record in measurement_records(part,phone_name,spec=source.evidence_kind=='spec'):
                if source.evidence_kind=='spec' and not record['role'].startswith('declared_'):continue
                verified.append(record)
    issues=[]
    for record in measurement_records(text,phone_name,games,claim=True):
        if record['owner']!=wanted or not any(record['role']==v['role'] and record['value']==v['value']
                and v['owner']==wanted and (not record['game'] or record['game']==v['game'])
                and (not record['role'].startswith(('fps_','device_temperature')) or
                     record['durations']<=v['durations'] and record['quality']<=v['quality'] and record['modes']==v['modes']) for v in verified):
            issues.append(record['match'][0])
    return issues

def measured_game_parts(quote):
    # Publishers frequently put the game/scenario in one sentence and the
    # measured value in the next. Keep that local context, reset at each game
    # or paragraph boundary, and never treat a frame-rate setting as a result.
    found = []
    game_name = r'《[^》]+》|原神|王者荣耀|星穹铁道|绝区零|和平精英'
    for paragraph in re.split(r' / |\n\s*\n', quote):
        context = ''
        for sentence in re.split(r'(?<=[。；])|\n', paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            if sentence.startswith('资料边界：'):
                break
            named = bool(re.search(game_name, sentence))
            if named:
                context = sentence
            combined = sentence if not context or context == sentence else context+' '+sentence
            numeric = re.search(r'\d+(?:\.\d+)?\s*(?:帧|fps|℃|°C|摄氏度|度|W|瓦|%|％)|平均帧率.{0,8}\d', sentence, re.I)
            conditions = re.search(r'\d+\s*(?:分钟|小时)|半小时|画质|跑图|折返跑|负载|场景|测试环境|最高帧率', combined)
            result = re.search(r'平均|实测(?:帧率|结果|达到|为)|测得|达到了|稳定在|跑到|耗电|功耗|温度|温升|(?:机身|背面|正面|表面).{0,6}(?:最高|峰值)', sentence)
            missing = re.search(r'没有|尚未|缺少|未做|不含|FAQ|相关阅读', sentence, re.I)
            if numeric and conditions and result and not missing:
                found.append(combined)
            elif context and not named and re.search(r'画质|模式|分钟|小时|场景|测试环境', sentence):
                context += ' '+sentence
    return list(dict.fromkeys(found))


def game_measurements(quote, game=None):
    """Coverage per requested measurement, scoped to a game and test conditions."""
    covered=set()
    for part in measured_game_parts(owned_fact_text(quote)):
        if game and not game_in_text(game,part):continue
        for clause in re.split(r'[，,。；;]',part):
            if re.search(r'没有|未(?:披露|记录|测|提供)|缺少|未知|待核|不含',clause):continue
            if re.search(r'(?:平均帧率|实测帧率)[^，。；]{0,12}\d+(?:\.\d+)?|(?:测得|稳定在|平均)[^，。；]{0,12}\d+(?:\.\d+)?\s*(?:帧|fps)',clause,re.I):covered.add('帧率')
            if re.search(r'(?:机身|背面|正面|表面|电池|核心|处理器|SoC|CPU|GPU|温升|峰值温度)[^，。；]{0,16}\d+(?:\.\d+)?\s*(?:℃|摄氏度|度)',clause,re.I):covered.add('温度')
            if re.search(r'(?:功耗|耗电功率)[^，。；]{0,12}\d+(?:\.\d+)?\s*(?:W|瓦)',clause,re.I):covered.add('功耗')
            if re.search(r'(?:耗电|电量下降|电量消耗)[^，。；]{0,12}\d+(?:\.\d+)?\s*[%％]',clause):covered.add('耗电')
    return covered


def _measured_durations(text):
    digits=dict(zip('零一二三四五六七八九',range(10))) | {'两':2}
    def number(value):
        if re.fullmatch(r'\d+(?:\.\d+)?',value):return Decimal(value)
        total,current=0,0
        for char in value:
            if char in {'十','百','千'}:
                total+=(current or 1)*{'十':10,'百':100,'千':1000}[char];current=0
            else:current=digits[char]
        return Decimal(total+current)
    return {(number(m[1]) if m[1] else Decimal(0))*(60 if m[3]=='小时' else 1)
            +(Decimal('0.5')*(60 if m[3]=='小时' else 1) if m[2] else 0)
            for m in re.finditer(r'((?:\d+(?:\.\d+)?|[零一二两三四五六七八九十百千]+))?(?:个)?(半)?\s*(分钟|小时)',text)
            if m[1] or m[2]}


def unsupported_game_averages(text, sources, games=()):
    # Keep each value attached to its own game and stated conditions. A union
    # of numbers from several games allowed exchanging their FPS unnoticed.
    pattern = re.compile(r'平均(?P<label>帧率)?\s*(?:约|为|达到(?:了)?|稳定在|在|：|:)?\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>帧|fps)?', re.I)
    names=canonical_games([*games,'原神','王者荣耀','崩坏：星穹铁道','绝区零','和平精英',
        *re.findall(r'《([^》]+)》',text+' '.join(s.quote for s in sources))])
    # canonical_games bounds user state; evidence may mention additional games.
    names=list(dict.fromkeys([*names,*games]))
    markers=[]
    for name in names:
        aliases=['崩坏：星穹铁道','崩坏:星穹铁道','星穹铁道','星铁'] if '星穹铁道' in name else [name]
        markers.extend((alias,name) for alias in aliases)
    def claims(value, *, implicit=False):
        for paragraph in re.split(r' / |\n\s*\n',value):
            active=()
            previous_end=0
            for match in pattern.finditer(paragraph):
                if not (match['label'] or match['unit']) and re.match(r'℃|度|[wW]|瓦|mAh|ms|毫秒|分钟|小时|%|元',paragraph[match.end():]):
                    continue
                prefix=paragraph[previous_end:match.start()]
                named=sorted((m.start(),-len(alias),name,m.end()) for alias,name in markers for m in re.finditer(re.escape(alias),prefix))
                if named:
                    # The last labelled game owns this measurement; multiple
                    # names joined by 和/与/、 refer to all of them.
                    last=named[-1]
                    active=(last[2],)
                    earlier=[n for n in named if n[0]<last[0] and re.fullmatch(r'[》\s]*(?:和|与|、|及)[《\s]*',prefix[n[3]:last[0]])]
                    if earlier:active=tuple(dict.fromkeys([n[2] for n in earlier]+[last[2]]))
                    context=prefix if len({n[2] for n in named})==1 else prefix[last[0]:]
                else:
                    context=prefix
                suffix=re.split(r'[。；;]',paragraph[match.end():])[0]
                next_names=[m.start() for alias,_ in markers for m in re.finditer(re.escape(alias),suffix)]
                if next_names:suffix=suffix[:min(next_names)]
                context+=' '+ ' '.join(part for part in re.split(r'[，,]',suffix)
                    if not re.search(r'不能|不代表|无法|尚无|未披露|缺少|没有',part))
                owners=active or (tuple(games) if implicit and len(games)==1 else ())
                durations=_measured_durations(context)
                quality=set(re.findall(r'(?:最高|极高|极低|高|中|低)画质',context))
                modes=set(m[0] for m in re.finditer(r'插帧|补帧|超帧|原生',context)
                    if not re.search(r'不是|关闭|未开启|非|未|不启用',context[max(0,m.start()-4):m.start()]))
                yield match,owners,durations,quality,modes
                previous_end=match.end()
    verified=[]
    for source in sources:
        if source.source_type=='snippet' or source.evidence_kind!='review':continue
        for part in measured_game_parts(source.quote):
            for match,owners,durations,quality,modes in claims(part):
                verified.extend((game,Decimal(match['value']),durations,quality,modes) for game in owners)
    issues=[]
    for match,owners,durations,quality,modes in claims(text,implicit=True):
        if not owners or any(not any(game==g and Decimal(match['value'])==v and durations<=d and quality<=q and modes==mode
                for g,v,d,q,mode in verified) for game in owners):
            issues.append(match[0])
    return issues

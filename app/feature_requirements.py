"""Finite, literal feature checks. Unsupported prose remains unknown.

Only body text establishes a feature. Negation, uncertainty and conflicting
records are kept separate from absence of evidence; no model judgment is used.
"""
from dataclasses import dataclass
import re
from typing import Literal


@dataclass(frozen=True)
class RequirementCheck:
    status: Literal['satisfied','unsatisfied','unknown','conflict']
    source_ids: tuple[str, ...] = ()
    urls: tuple[str, ...] = ()

    @property
    def label(self):
        return {'satisfied':'配置符合', 'unsatisfied':'配置不符合',
                'unknown':'待逐项核对', 'conflict':'来源冲突，待核对'}[self.status]


# Canonical property -> spellings meaning True / False. This is a documented
# capability set, not a promise to understand arbitrary natural-language gates.
FEATURES = {
    'curved_screen': (r'曲面屏|曲屏|曲面显示屏|微曲屏|等深微曲屏', r'直屏|平面屏|平直屏幕'),
    'wireless_charging': (r'无线充电', None),
    'wireless_fast_charging': (r'无线快充|无线闪充', None),
    'nfc': (r'(?<![a-z])NFC(?![a-z])', None),
    'headphone_jack': (r'3[.．]5\s*mm耳机(?:孔|接口)|耳机插孔', None),
}
RELATION = r'是|为|采用|使用|支持|具备|配备|提供|带有|具有|拥有|含有|带|含'
NEGATIVE = r'(?:不|未|没)(?:'+RELATION+r')|不需要|不想要|不接受|不要|没有|无法|无须|无需|取消|排除|不能有|不得有|非|无'
CANCEL = r'不要求|不强求|无所谓|不限|都行|取消.*(?:要求|限制)|不再需要|不需要'
UNCERTAIN = r'待确认|待核对|未知|未确认|不确定|可能|是否|尚未公布|未提及|未说明|没有说明|没有确认|据说|传闻|不是不|并非不|不能不|不要不|[?？]'


def clauses(text):
    return re.split(r'[，,。；;\n]|但是|不过|而且', text)


def feature_values(text, *, demand=False):
    """Return canonical property/value pairs with locally scoped negation."""
    found=[]
    for clause in clauses(text):
        if re.search(r'(?:资料|记录|正文|页面).{0,8}(?:没有|未|缺)|(?:没有|缺少|未提供|未查到).{0,18}(?:描述|资料|记录|说明|证据)',clause):continue
        if re.search(UNCERTAIN, clause, re.I) or demand and re.search(CANCEL, clause):
            continue
        for key, spellings in FEATURES.items():
            for positive, pattern in zip((True, False), spellings):
                if not pattern:
                    continue
                for match in re.finditer(pattern, clause, re.I):
                    before=clause[max(0,match.start()-12):match.start()]
                    after=clause[match.end():match.end()+12]
                    if key in {'wireless_charging','wireless_fast_charging'} and before.endswith('反向'):
                        continue  # Transmitting power does not prove receiving it.
                    # A mention alone in a specification table is not support.
                    negative=bool(re.search('(?:'+NEGATIVE+r')(?:采用|使用|支持|配备|的|是|为)?\s*(?:\d+\s*W)?\s*$', before,re.I)
                        or re.match(r'\s*[:：]?\s*(?:不支持|不具备|无|没有|不需要|不要|不接受|否)', after))
                    affirmative=bool(re.search(RELATION+r'|搭载|需要|必须|要求|只接受|想要|要',before)
                        or re.match(r'\s*[:：]?\s*(?:支持|具备|有|是|(?:功率)?\s*\d+\s*W)',after,re.I)
                        or key=='curved_screen')
                    if negative and (re.search(r'\d+\s*W\s*$',before,re.I)
                            or re.match(r'\s*[:：]?\s*不支持\s*\d+\s*W',after,re.I)):
                        continue  # Rejecting one wattage does not reject all charging.
                    if negative or affirmative:
                        value=not positive if negative else positive
                        found.append((key,value))
                        # Fast charging implies charging; lack of fast charging
                        # does not imply lack of charging (and vice versa).
                        if key=='wireless_fast_charging' and value:
                            found.append(('wireless_charging',True))
                        elif key=='wireless_charging' and not value:
                            found.append(('wireless_fast_charging',False))
    return list(dict.fromkeys(found))


def feature_keys(text):
    return {key for key, patterns in FEATURES.items() if any(p and re.search(p,text,re.I) for p in patterns)}


def feature_observations(sources, registry=()):
    from .evidence_rules import owned_fact_text
    sources=[s.model_copy(update={'quote':owned_fact_text(s.quote)}) for s in sources]
    registry=[s.model_copy(update={'quote':owned_fact_text(s.quote)}) for s in registry]
    facts=[(key,value,s) for s in sources if s.source_type!='snippet' for key,value in feature_values(s.quote)]
    newest={}
    for s in registry or sources:
        if s.source_type=='snippet':continue
        for key,value in feature_values(s.quote):
            identity=(s.phone_id,s.url,key)
            newest[identity]=max(newest.get(identity,''),s.checked_on)
    return [(k,v,s) for k,v,s in facts if s.checked_on>=newest.get((s.phone_id,s.url,k),'')]


def feature_check(requirement, sources):
    wanted=feature_values(requirement,demand=True)
    if not wanted:
        return RequirementCheck('unknown')
    results=[]; used=[]
    for key,value in wanted:
        targets={v for k,v in wanted if k==key}
        hits=[(v,s) for k,v,s in feature_observations(sources) if k==key]
        values={v for v,_ in hits}
        used.extend(s for _,s in hits)
        results.append('conflict' if len(targets)>1 or len(values)>1 else
            'unknown' if not values else 'satisfied' if value in values else 'unsatisfied')
    status=next((v for v in ('conflict','unsatisfied','unknown') if v in results),'satisfied')
    return RequirementCheck(status,tuple(dict.fromkeys(s.id for s in used)),tuple(dict.fromkeys(s.url for s in used)))


class RequirementOverflow(ValueError):
    pass


def merge_requirements(existing, additions):
    """Replace only the addressed property; same-turn contradictions survive."""
    old=[a for r in existing for a in requirement_atoms(r)]
    new=[a for r in additions for a in requirement_atoms(r)]
    keys={a.key for a in new}
    result=list(dict.fromkeys([a.text for a in old if a.key not in keys]+[a.text for a in new]))
    if len(result)>12:raise RequirementOverflow('具体要求最多保留12项，请先明确取消不再需要的要求；原条件已保留。')
    return result


def literal_feature_updates(profile, text):
    profile.requirements=update_requirements(profile.requirements,text)


@dataclass(frozen=True)
class RequirementAtom:
    """Stable property identity plus its original span; strings remain API compatible."""
    key: str
    text: str
    start: int
    end: int
    operator: str = ''
    value: str = ''
    unit: str = ''


_DIMENSIONS={'weight':r'重量|机身重|克|\bg\b','thickness':r'厚度|毫米|\bmm\b',
    'storage':r'\b(?:GB|TB)\b|\d\s*(?:GB|TB)|存储|容量版本',
    'refresh_rate':r'刷新率|赫兹|\bHz\b','battery':r'电池|mAh'}
_CONTROL=r'联网|网页查询|本地资料|已有资料|测试条件|有.*条件的资料|引用|来源编号|保留.*日期|到手价'
_REMOVE=r'取消|不再(?:需要|要求|限制)|不限制|不限|不要求|不强求|无所谓|都行|去掉|删掉'
_ALL_REMOVE=r'(?:取消|清空|去掉).{0,5}(?:所有|全部|全部的)(?:具体)?(?:要求|条件|限制)|(?:要求|条件)(?:全部|都)清空'
REPORT_FIELDS={'测试条件':r'测试条件|有[^，。；]{0,8}条件的资料',
    '核对日期':r'(?:核对|查验|记录)(?:的)?(?:日期|时间)|保留[^，。；]{0,8}日期|原日期',
    '存储版本':r'存储版本|具体版本|容量版本','来源引用':r'来源引用|引用来源|附上来源'}


def reporting_requirement(text):
    """Classify the operation before its nouns (storage/date are not predicates)."""
    if re.search(r'(?:不止(?:是)?|不限于|不要|不考虑|排除|取消|重新考虑)\s*(?:[A-Za-z\u4e00-\u9fff]+\s*)?\d+\s*系列(?:的(?:手机|机型))?[。！!]*$',text):return True
    bounded=bool(re.search(r'至少|不超过|不低于|不高于|最多|[<>]=?|[≥≤]',text) and re.search(r'\d',text))
    capability=bool(feature_values(text,demand=True))
    strong=bool(re.search(r'必须|只接受|硬性|不得|不能|需要',text))
    report_object=any(re.search(p,text) for p in REPORT_FIELDS.values())
    report_verb=bool(re.match(r'(?:请|需要|要求)?(?:说明|告诉|列出|给出|展示|标明|注明|保留|引用|解释|核对)',text))
    if report_verb:return True
    if bounded or capability:return False
    if strong and re.search(r'\d+\s*(?:GB|TB|mAh|Hz|毫米|mm|克|g)(?![a-z])',text,re.I):return False
    if report_object:return True
    return bool(re.search(_CONTROL,text)) and not strong


def requirement_atoms(text):
    """Split only condition conjunctions, preserving unknown product gates verbatim.

    These spans are never treated as proof about a phone. Unsupported conditions
    remain independent unknowns instead of being discarded or attached to storage.
    """
    parts=re.finditer(r'[^，,。；;、\n]+',text)
    atoms=[]
    for fragment in parts:
        separators=list(re.finditer(r'并且|而且|以及|同时|且|和(?!平精英)|并(?=说明|展示|保留|注明|列出|给出|告诉)',fragment[0]))
        starts=[0,*[m.end() for m in separators]]
        ends=[*[m.start() for m in separators],len(fragment[0])]
        for left,right in zip(starts,ends):
            raw=fragment[0][left:right]
            value=raw.strip()
            if not value:continue
            if reporting_requirement(value):continue
            keys=feature_keys(value)
            numeric=next((k for k,pattern in _DIMENSIONS.items() if re.search(pattern,value,re.I)),None)
            # Operations and evidence/permission instructions are not purchase facts.
            if re.search(_REMOVE,value) or re.search(_CONTROL,value) and not keys and not numeric:continue
            if re.search(r'^(?:按|可以|请|重新|推荐|预算|主要看|主要玩)',value) and not keys and not numeric and not re.search(r'必须|需要|只接受|硬性|至少|不超过',value):continue
            value=re.sub(r'(?:都)?(?:保留|不变|继续沿用)[。！!]*$','',value).strip()
            if not value:continue
            span_value=value
            if left and not keys and not numeric and re.search(r'必须|需要|只接受|硬性',fragment[0]):
                value='需要'+value
            key=('wireless_fast_charging' if 'wireless_fast_charging' in keys else
                 sorted(keys)[0] if keys else numeric or 'text:'+re.sub(r'\W','',value).casefold())
            number=re.search(r'(\d+(?:\.\d+)?)\s*(GB|TB|mAh|Hz|mm|毫米|克|g)(?![a-z])',value,re.I) if numeric else None
            op='min' if re.search(r'至少|不低于|起码|>=|≥|以上',value) else 'max' if re.search(r'不超过|不高于|最多|至多|<=|≤|以下',value) else ''
            start=fragment.start()+left+raw.find(span_value)
            atoms.append(RequirementAtom(key,value,start,start+len(span_value),op,number[1] if number else '',number[2].lower() if number else ''))
    return atoms


def requirement_key(text):
    atoms=requirement_atoms(text)
    return atoms[0].key if len(atoms)==1 else None


def field_keys(text):
    return feature_keys(text)|{k for k,p in _DIMENSIONS.items() if re.search(p,text,re.I)}


def observed_fields(text):
    from .evidence_rules import parameter_values
    units={'g':'weight','mm':'thickness','gb':'storage','hz':'refresh_rate','mah':'battery'}
    return {k for k,v in feature_values(text)}|{units[u] for v,u,r,m in parameter_values(text)
        if u in units and (u!='hz' or r=='display_refresh') and (u!='mm' or r=='thickness')}


def _signature(atom):
    return (atom.key,atom.operator,atom.value,atom.unit,tuple(feature_values(atom.text,demand=True)))


def update_requirements(existing,text,*,additions=(),removals=(),clear=False):
    """One reducer for model and offline updates; unmentioned properties survive."""
    old=[a for r in existing for a in requirement_atoms(r)]
    cancel_keys=set(); literal=[]
    operation_clauses=re.split(r'[，,。；;、\n]|但是|不过|但|(?=(?:并且|并|而且|而|同时|然后)(?:取消|保留|维持|改为|改成|不限制|需要|要求))',text)
    for clause in operation_clauses:
        if re.search(_REMOVE,clause):
            target=re.split(r'保留|维持|不变|继续沿用',clause)[0]
            cancel_keys.update(feature_keys(target))
            cancel_keys.update(k for k,p in _DIMENSIONS.items() if re.search(p,target,re.I))
        elif not re.search(UNCERTAIN,clause,re.I) and not re.search(r'保留|不变',clause):
            for atom in requirement_atoms(clause):
                if (feature_keys(atom.text) and re.search(r'需要|要求|必须|不要|不接受|只接受|想要|改成|改为|排除|不能有|支持',atom.text)
                    or atom.operator and atom.value or (atom.value or atom.key.startswith('text:')) and re.search(r'必须|硬性|只接受|需要',atom.text)):literal.append(atom.text)
    grounded=[]; current=requirement_atoms(text)
    for addition in additions:
        for atom in requirement_atoms(addition):
            if atom.text in text:
                grounded.append(atom.text)
            else:
                equivalent=next((a for a in current if _signature(a)==_signature(atom)),None)
                if equivalent:grounded.append(equivalent.text)
    if additions:
        for clause in clauses(text):
            if re.search(_CONTROL+'|'+_REMOVE+'|'+UNCERTAIN+r'|怎么样|如何',clause):continue
            habit=bool(re.search(r'玩|游戏|拍|摄影|导航|刷视频',clause) and re.search(
                r'小时|分钟|常|主要|喜欢|家里|每次|连续',clause))
            if habit and not any(a.operator or feature_keys(a.text) for a in requirement_atoms(clause)):
                grounded.extend(a.text for a in requirement_atoms(clause))
    # A model may refer to an old compound. Only cancel its explicitly addressed atoms.
    remove_text=set()
    if re.search(_REMOVE,text):
        for removal in removals:
            for atom in requirement_atoms(removal):
                if atom.key in cancel_keys:remove_text.add(atom.text)
                elif atom.key.startswith('text:') and (atom.text in text or len(old)==1 and re.search(r'这个|该|那个',text)):
                    remove_text.add(atom.text)
        if len(old)==1 and removals and re.search(r'取消这个限制|取消那个要求',text):remove_text.add(old[0].text)
    if clear and re.search(_ALL_REMOVE,text):old=[]
    old=[a.text for a in old if a.key not in cancel_keys and a.text not in remove_text]
    return merge_requirements(old,[*grounded,*literal])

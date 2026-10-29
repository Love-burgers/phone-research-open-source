"""Bounded working evidence: marginal coverage, provenance and recency."""
from .feature_requirements import feature_values
from .selection_requirements import check_requirement, topic_observations
from .selection_presentation import configuration_facts
from .selection_prices import source_price_options


def source_roles(source, phone, profile, focus_topics=()):
    if source.source_type=='snippet':
        return set()
    roles={'body'}
    if source.market=='中国大陆' and source_price_options(source):
        # Different prices for the same version must remain visible as conflict.
        roles.update(('price',offer.variant,offer.price) for offer in source_price_options(source))
    for requirement in profile.requirements:
        check=check_requirement(requirement,[source])
        if check.status!='unknown':
            roles.add(('requirement',requirement,check.status))
    for topic in set(profile.priorities)|set(focus_topics):
        if configuration_facts(phone,[topic],[source]):roles.add(('configuration',topic))
        if topic_observations(phone.id,topic,[source],profile.games):roles.add(('observation',topic))
    # Retain opposing literal feature statements even when one compound source
    # contains both; the requirement checker must see the conflict.
    roles.update(('feature',k,v) for k,v in feature_values(source.quote)
                 if any(k==key for r in profile.requirements for key,_ in feature_values(r,demand=True)))
    return roles


def retained_sources(sources, phone, profile, *, limit=8, focus_topics=(), protected=()):
    kept=list(sources)
    roles={s.id:source_roles(s,phone,profile,focus_topics) for s in kept}
    weights={'price':7,'requirement':9,'feature':9,'observation':5,'configuration':4}
    def marginal(source):
        others=set().union(*(roles[s.id] for s in kept if s.id!=source.id))
        loss=sum(weights.get(role[0],1) if isinstance(role,tuple) else 1
                 for role in roles[source.id]-others)
        return (source.id in protected,loss,source.source_type!='snippet',
                source.checked_on,source.published_on or '',kept.index(source))
    while len(kept)>limit:
        removable=[s for s in kept if s.id not in protected]
        if not removable:
            break
        kept.remove(min(removable,key=marginal))
    return kept

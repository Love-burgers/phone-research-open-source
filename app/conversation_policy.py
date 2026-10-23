"""Route a validated demand delta, independently of recommendation generation."""
from dataclasses import dataclass
from .selection_requirements import selection_threshold


@dataclass(frozen=True)
class ConversationRoute:
    operation: str
    changed_fields: tuple[str, ...]
    focus: str
    reason: str

    @property
    def recommends(self):
        return self.operation in {"recommend", "revise", "alternatives"}


def route_turn(previous, current, intent, *, has_candidates, selected, incompatible=False):
    # Operational candidate/focus fields are not user constraints. Classify the
    # validated delta before candidate selection mutates those fields.
    changed = tuple(field for field in type(current).model_fields
                    if field not in {"candidate_ids", "explicit_candidate_ids", "focus_phone_id"}
                    and previous[field] != getattr(current, field))
    focus = intent.response_focus
    if not has_candidates:
        return ConversationRoute("recommend", changed, focus, "no_current_candidates")
    if intent.candidate_action == "alternatives":
        return ConversationRoute("alternatives", changed, focus, "explicit_alternatives")
    if (set(changed) == {'games'} and set(previous['games']) <= set(current.games)
            and intent.turn_action == 'revise' and not intent.reselect and not incompatible):
        # Naming another workload is a detail of the existing gaming use. A
        # generic model 'revise' label alone must not replace the shortlist.
        return ConversationRoute('refine', changed, focus, 'additive_workload_without_reselection')
    if intent.reselect or intent.turn_action in {"recommend", "revise"}:
        return ConversationRoute("revise", changed, focus, "explicit_selection_update")
    # A semantic 'refine' cannot bypass changed budget/brands/purpose or a
    # demonstrated conflict. Unknown evidence is not a demonstrated conflict.
    details = {"games", "requirements"}
    removed_detail = any(set(previous[field]) - set(getattr(current, field)) for field in details)
    added_gates = any(selection_threshold(item) for item in current.requirements
                      if item not in previous['requirements'])
    if incompatible or set(changed) - details or removed_detail or added_gates:
        return ConversationRoute("revise", changed, focus, "constraint_delta_or_known_conflict")
    if selected and intent.turn_action=='question':
        return ConversationRoute('question',changed,focus,'referenced_candidate_detail')
    if selected and not changed:
        return ConversationRoute("question", changed, focus, "referenced_candidate")
    if changed and (intent.turn_action == "refine" or set(changed) <= {"games"}):
        # Older/offline extractors have no semantic action. Only their structured
        # game-name additions have an unambiguous detail role; arbitrary unknown
        # requirements are conservatively rechecked, never guessed as harmless.
        return ConversationRoute("refine", changed, focus, "additive_detail")
    if changed:
        return ConversationRoute("revise", changed, focus, "unclassified_requirement_delta")
    if intent.turn_action == "refine":
        return ConversationRoute("refine", changed, focus, "contextual_detail")
    return ConversationRoute("question", changed, focus, "continue_current_question")

# salespath_env/server/rules.py

from dataclasses import dataclass
from typing import Callable

from ..models import SalesPathAction, SalesPathState


@dataclass
class BusinessRule:
    """
    Returns True when the rule is VIOLATED.
    """

    rule_id: str
    name: str
    description: str
    check: Callable[[SalesPathState, SalesPathAction], bool]


def _qualify_before_present(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R01:
    PRESENT before QUALIFY is invalid.
    """
    if action.action_type == "PRESENT":
        return "QUALIFY" not in state.steps_completed
    return False


def _demo_before_negotiate(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R02:
    NEGOTIATE before OFFER_DEMO is invalid.
    """
    if action.action_type == "NEGOTIATE":
        return "OFFER_DEMO" not in state.steps_completed
    return False


def _budget_known_to_negotiate(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R03:
    Cannot NEGOTIATE while budget is unknown.
    """
    if action.action_type == "NEGOTIATE":
        return state.prospect_profile.get("budget_signal") == "unknown"
    return False


def _discount_after_objections(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R04:
    Discount only after 2 objections handled.
    """
    if action.action_type == "NEGOTIATE":
        if "discount" in action.content.lower():
            return state.objections_handled < 2
    return False


def _no_repeat_action(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R05:
    Same action twice in a row is invalid.
    """
    if state.conversation_history:
        last_action = state.conversation_history[-1].get("action_type", "")
        return last_action == action.action_type
    return False


def _prospect_first(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R06:
    First action must be PROSPECT.
    """
    if state.turn_number == 1:
        return action.action_type != "PROSPECT"
    return False


def _followup_timing(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R07:
    FOLLOW_UP only valid after silence.
    If prospect just responded last turn, violation.
    """
    if action.action_type == "FOLLOW_UP":
        if state.conversation_history:
            last_speaker = state.conversation_history[-1].get("speaker", "agent")
            return last_speaker == "prospect"
    return False


def _disqualify_logic(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R08:
    DISQUALIFY only when prospect is genuinely not closeable.
    Violation if prospect is actually closeable.
    """
    if action.action_type == "DISQUALIFY":
        true_budget = state.hidden_state.get("true_budget", 0.5)
        close_threshold = state.hidden_state.get("close_threshold", 0.5)
        decision_maker = state.prospect_profile.get("decision_maker", True)

        return (true_budget >= close_threshold) and decision_maker

    return False


def _close_requires_demo(
    state: SalesPathState,
    action: SalesPathAction,
) -> bool:
    """
    R09:
    Difficulty 2+ requires OFFER_DEMO before CLOSE.
    """
    if action.action_type == "CLOSE":
        if state.difficulty >= 2:
            return "OFFER_DEMO" not in state.steps_completed
    return False


BUSINESS_RULES = [
    BusinessRule(
        "R01",
        "qualify_before_present",
        "Must QUALIFY before PRESENT",
        _qualify_before_present,
    ),
    BusinessRule(
        "R02",
        "demo_before_negotiate",
        "Must OFFER_DEMO before NEGOTIATE",
        _demo_before_negotiate,
    ),
    BusinessRule(
        "R03",
        "budget_known_to_negotiate",
        "Budget must be known before NEGOTIATE",
        _budget_known_to_negotiate,
    ),
    BusinessRule(
        "R04",
        "discount_after_objections",
        "Discount only after 2 objections handled",
        _discount_after_objections,
    ),
    BusinessRule(
        "R05",
        "no_repeat_action",
        "Cannot repeat same action consecutively",
        _no_repeat_action,
    ),
    BusinessRule(
        "R06",
        "prospect_first",
        "First action must be PROSPECT",
        _prospect_first,
    ),
    BusinessRule(
        "R07",
        "followup_timing",
        "FOLLOW_UP only after prospect silence",
        _followup_timing,
    ),
    BusinessRule(
        "R08",
        "disqualify_logic",
        "DISQUALIFY only when prospect is genuinely unqualified",
        _disqualify_logic,
    ),
    BusinessRule(
        "R09",
        "close_requires_demo",
        "Must OFFER_DEMO before CLOSE (difficulty 2+)",
        _close_requires_demo,
    ),
]


def check_rules(
    state: SalesPathState,
    action: SalesPathAction,
) -> list[str]:
    """
    Returns list of violated rule IDs.
    """

    violated = []

    for rule in BUSINESS_RULES:
        if rule.check(state, action):
            violated.append(rule.rule_id)

    return violated
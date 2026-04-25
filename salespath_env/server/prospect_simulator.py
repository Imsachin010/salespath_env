# salespath_env/server/prospect_simulator.py

from ..models import SalesPathAction, SalesPathState


RESPONSE_TEXT = {
    "open:positive_signal": "That sounds interesting. Tell me more about how this works.",
    "open:neutral_signal": "I see. We're evaluating a few options at the moment.",

    "objection:price": "The pricing seems higher than what we budgeted for.",
    "objection:timing": "The timing isn't ideal — we're in the middle of a quarter close.",
    "objection:premature_pitch": (
        "I'm not sure we're ready to discuss solutions yet. "
        "What do you know about our current situation?"
    ),

    "deflect:budget_not_discussed": (
        "We haven't really talked about what we're looking for yet."
    ),
    "deflect:stall": (
        "Let me get back to you on this. A lot is happening on our end."
    ),

    "accept:demo_scheduled": (
        "Yes, let's set up a demo. What time works next week?"
    ),
    "accept:close_success": (
        "Alright, I think we can move forward with this. "
        "Send over the paperwork."
    ),

    "reject:close_failed": (
        "I don't think we're ready to commit at this point."
    ),

    "silence": "",

    "exit:disqualified": (
        "I think we're done here. This isn't the right fit."
    ),
}


class ProspectSimulator:
    """
    Pure rule-based simulator.
    No LLM. No transformers. Deterministic behavior.
    """

    def respond(
        self,
        action: SalesPathAction,
        state: SalesPathState,
    ) -> tuple[str, str]:
        """
        Returns:
            (response_token, response_text)
        """

        token = self._get_token(action, state)
        text = RESPONSE_TEXT[token]

        return token, text

    def _get_token(
        self,
        action: SalesPathAction,
        state: SalesPathState,
    ) -> str:
        atype = action.action_type
        difficulty = state.difficulty
        turn = state.turn_number
        profile = state.prospect_profile
        hidden = state.hidden_state
        objections = state.objections_handled

        # -----------------------------
        # Rule-triggered responses first
        # -----------------------------

        if state.constraints_violated:
            latest = state.constraints_violated[-1]

            if latest == "R01":
                return "objection:premature_pitch"

            if latest == "R03":
                return "deflect:budget_not_discussed"

        # -----------------------------
        # Action-based responses
        # -----------------------------

        if atype == "PROSPECT":
            return "open:positive_signal"

        if atype == "QUALIFY":
            # Reveal budget if hidden
            if profile.get("budget_signal") == "unknown":
                state.prospect_profile["budget_signal"] = hidden.get(
                    "revealed_budget",
                    "medium",
                )

            return "open:neutral_signal"

        if atype == "PRESENT":
            if difficulty >= 2:
                if objections == 0:
                    return "objection:price"

            return "open:positive_signal"

        if atype == "HANDLE_OBJECTION":
            state.objections_handled += 1

            required_objections = hidden.get("num_objections", 1)

            if state.objections_handled >= required_objections:
                return "open:positive_signal"

            if objections == 0:
                return "objection:timing"

            return "open:positive_signal"

        if atype == "OFFER_DEMO":
            return "accept:demo_scheduled"

        if atype == "NEGOTIATE":
            return "open:neutral_signal"

        if atype == "CLOSE":
            true_budget = hidden.get("true_budget", 0.7)
            close_threshold = hidden.get("close_threshold", 0.5)
            decision_maker = profile.get("decision_maker", True)

            if (
                true_budget >= close_threshold
                and decision_maker
            ):
                return "accept:close_success"

            return "reject:close_failed"

        if atype == "FOLLOW_UP":
            return "open:neutral_signal"

        if atype == "DISQUALIFY":
            return "exit:disqualified"

        # -----------------------------
        # Difficulty 3+ mode shift
        # -----------------------------

        if difficulty >= 3 and turn >= 10:
            import random

            if random.random() < hidden.get("stall_probability", 0.0):
                return "deflect:stall"

        return "open:neutral_signal"
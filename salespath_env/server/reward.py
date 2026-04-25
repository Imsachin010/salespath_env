# salespath_env/server/reward.py

from ..models import SalesPathAction, SalesPathState


DIFFICULTY_OPTIMAL_TURNS = {
    1: 5,
    2: 8,
    3: 12,
    4: 14,
}


def compute_reward(
    state: SalesPathState,
    action: SalesPathAction,
    response_token: str,
    new_violations: list[str],
    episode_done: bool,
) -> tuple[float, dict]:
    """
    Returns:
        (total_reward, reward_components)
    """

    components = {}

    # --------------------------------------------------
    # 1. Outcome Reward (terminal only)
    # --------------------------------------------------

    r_outcome = 0.0

    if episode_done:
        if response_token == "accept:close_success":
            r_outcome = 1.0

        elif action.action_type == "DISQUALIFY":
            if "R08" not in new_violations:
                r_outcome = 0.5
            else:
                r_outcome = -0.5

        elif state.turn_number >= 20:
            r_outcome = -0.3

        elif len(state.constraints_violated) >= 3:
            r_outcome = -0.5

        else:
            r_outcome = -0.5

    components["r_outcome"] = r_outcome

    # --------------------------------------------------
    # 2. Compliance Reward
    # --------------------------------------------------

    r_compliance = max(
        -1.0,
        -0.2 * len(new_violations),
    )

    components["r_compliance"] = r_compliance

    # --------------------------------------------------
    # 3. Ordering Reward
    # --------------------------------------------------

    required = state.required_workflow
    completed = state.steps_completed

    if len(required) > 0 and len(completed) > 0:
        correct = sum(
            1
            for i in range(min(len(required), len(completed)))
            if required[i] == completed[i]
        )

        r_ordering = correct / len(required)

    else:
        r_ordering = 1.0

    components["r_ordering"] = r_ordering

    # --------------------------------------------------
    # 4. Efficiency Reward
    # --------------------------------------------------

    if episode_done:
        optimal = DIFFICULTY_OPTIMAL_TURNS.get(
            state.difficulty,
            10,
        )

        extra_turns = max(
            0,
            state.turn_number - optimal,
        )

        r_efficiency = max(
            -0.3,
            -0.05 * extra_turns,
        )

    else:
        r_efficiency = 0.0

    components["r_efficiency"] = r_efficiency

    # --------------------------------------------------
    # 5. Format Reward
    # --------------------------------------------------

    r_format = 1.0 if action.is_valid() else -0.1
    components["r_format"] = r_format

    # --------------------------------------------------
    # Final Weighted Reward
    # --------------------------------------------------

    weights = {
        "r_outcome": 0.40,
        "r_compliance": 0.30,
        "r_ordering": 0.15,
        "r_efficiency": 0.10,
        "r_format": 0.05,
    }

    total_reward = sum(
        weights[key] * components[key]
        for key in weights
    )

    components["total"] = total_reward

    return total_reward, components
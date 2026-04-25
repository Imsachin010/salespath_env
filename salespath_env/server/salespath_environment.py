# salespath_env/server/salespath_environment.py

import uuid

from openenv.core.env_server import Environment

from ..models import (
    SalesPathAction,
    SalesPathObservation,
    SalesPathState,
)
from .task_bank import sample_profile
from .rules import check_rules
from .reward import compute_reward
from .prospect_simulator import ProspectSimulator


DIFFICULTY_WORKFLOW = {
    1: [
        "QUALIFY",
        "PRESENT",
        "CLOSE",
    ],
    2: [
        "QUALIFY",
        "PRESENT",
        "HANDLE_OBJECTION",
        "OFFER_DEMO",
        "CLOSE",
    ],
    3: [
        "QUALIFY",
        "PRESENT",
        "HANDLE_OBJECTION",
        "OFFER_DEMO",
        "HANDLE_OBJECTION",
        "NEGOTIATE",
        "CLOSE",
    ],
    4: [],  # Agent must determine; DISQUALIFY may be correct
}


MAX_VIOLATIONS_BEFORE_TERMINATE = 3
MAX_TURNS = 20


class SalesPathEnvironment(Environment):
    """
    Core OpenEnv environment.
    All business logic routes through:
    - rules.py
    - reward.py
    - prospect_simulator.py
    """

    def __init__(self):
        super().__init__()
        self._state = SalesPathState()
        self._simulator = ProspectSimulator()

    def reset(self, difficulty: int = 1) -> SalesPathObservation:
        """
        Start a new episode.
        """

        profile = sample_profile(difficulty)

        hidden_state = {
            "true_budget": profile.true_budget,
            "close_threshold": profile.close_threshold,
            "stall_probability": profile.stall_probability,
            "num_objections": {
                1: 0,
                2: 1,
                3: 2,
                4: 2,
            }[difficulty],
            "revealed_budget": (
                "high"
                if profile.true_budget >= 0.7
                else "medium"
                if profile.true_budget >= 0.4
                else "low"
            ),
        }

        public_profile = {
            "company_name": profile.company_name,
            "company_size": profile.company_size,
            "industry": profile.industry,
            "budget_signal": profile.budget_signal,
            "pain_points": profile.pain_points,
            "decision_maker": profile.decision_maker,
        }

        self._state = SalesPathState(
            episode_id=str(uuid.uuid4()),
            prospect_profile=public_profile,
            conversation_history=[],
            workflow_stage="START",
            required_workflow=DIFFICULTY_WORKFLOW[difficulty],
            steps_completed=[],
            constraints_violated=[],
            objections_handled=0,
            turn_number=0,
            difficulty=difficulty,
            done=False,
            hidden_state=hidden_state,
        )

        intro_message = (
            f"You are engaging {profile.company_name}, "
            f"a {profile.company_size} {profile.industry} company. "
            f"Pain points: {', '.join(profile.pain_points)}. "
            f"Begin the sales conversation."
        )

        return SalesPathObservation(
            prospect_response=intro_message,
            workflow_stage="START",
            constraints_violated=[],
            steps_completed=[],
            turn_number=0,
            reward=0.0,
            reward_components={},
            done=False,
            info={
                "difficulty": difficulty,
                "episode_id": self._state.episode_id,
            },
        )

    def step(
        self,
        action: SalesPathAction,
    ) -> SalesPathObservation:
        """
        One environment transition.
        """

        state = self._state

        # -----------------------------------
        # Advance turn
        # -----------------------------------

        state.turn_number += 1

        # -----------------------------------
        # Strict action validation
        # Must return observation, never crash
        # -----------------------------------

        if not action.is_valid():
            return SalesPathObservation(
                prospect_response="Invalid action type.",
                workflow_stage=state.workflow_stage,
                constraints_violated=list(state.constraints_violated),
                steps_completed=list(state.steps_completed),
                turn_number=state.turn_number,
                reward=-0.2,
                reward_components={
                    "r_format": -0.1,
                },
                done=False,
                info={
                    "error": (
                        f"Invalid action_type: "
                        f"{action.action_type}"
                    )
                },
            )

        # -----------------------------------
        # Rule checks
        # -----------------------------------

        new_violations = check_rules(
            state,
            action,
        )

        state.constraints_violated.extend(
            new_violations
        )

        # -----------------------------------
        # Record agent action
        # -----------------------------------

        state.conversation_history.append(
            {
                "turn": state.turn_number,
                "speaker": "agent",
                "action_type": action.action_type,
                "content": action.content,
            }
        )

        # -----------------------------------
        # Update workflow state
        # -----------------------------------

        if action.action_type not in state.steps_completed:
            state.steps_completed.append(
                action.action_type
            )

        state.workflow_stage = action.action_type

        # -----------------------------------
        # Prospect response
        # -----------------------------------

        response_token, response_text = (
            self._simulator.respond(
                action,
                state,
            )
        )

        state.conversation_history.append(
            {
                "turn": state.turn_number,
                "speaker": "prospect",
                "response_token": response_token,
                "text": response_text,
            }
        )

        # -----------------------------------
        # Episode termination
        # -----------------------------------

        terminal_actions = {
            "CLOSE",
            "DISQUALIFY",
        }

        too_many_violations = (
            len(state.constraints_violated)
            >= MAX_VIOLATIONS_BEFORE_TERMINATE
        )

        turn_limit_reached = (
            state.turn_number >= MAX_TURNS
        )

        done = (
            action.action_type in terminal_actions
            or too_many_violations
            or turn_limit_reached
        )

        state.done = done

        # -----------------------------------
        # Reward
        # -----------------------------------

        total_reward, components = (
            compute_reward(
                state=state,
                action=action,
                response_token=response_token,
                new_violations=new_violations,
                episode_done=done,
            )
        )

        return SalesPathObservation(
            prospect_response=response_text,
            workflow_stage=state.workflow_stage,
            constraints_violated=list(
                state.constraints_violated
            ),
            steps_completed=list(
                state.steps_completed
            ),
            turn_number=state.turn_number,
            reward=total_reward,
            reward_components=components,
            done=done,
            info={
                "response_token": response_token,
                "new_violations": new_violations,
                "episode_id": state.episode_id,
            },
        )

    @property
    def state(self) -> SalesPathState:
        return self._state
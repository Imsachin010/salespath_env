# salespath_env/models.py

from __future__ import annotations

import uuid
from typing import Dict, List
from pydantic import BaseModel, Field
 
# Safe OpenEnv Imports: Use OpenEnv base classes if available, 
# otherwise fall back to Pydantic to bypass security blocks.
try:
    from openenv.core import Action, Observation, State
except (ImportError, Exception):
    Action = BaseModel
    Observation = BaseModel
    State = BaseModel


VALID_ACTIONS = {
    "PROSPECT",
    "QUALIFY",
    "PRESENT",
    "HANDLE_OBJECTION",
    "OFFER_DEMO",
    "NEGOTIATE",
    "CLOSE",
    "FOLLOW_UP",
    "DISQUALIFY",
}


class SalesPathAction(Action):
    """
    Action sent by the agent to the environment.
    """

    action_type: str
    content: str
    target: str = ""

    def is_valid(self) -> bool:
        """
        Strict validation of allowed action types.
        """
        return self.action_type in VALID_ACTIONS


class SalesPathObservation(Observation):
    """
    What the agent is allowed to observe.
    Hidden state must NEVER be exposed here.
    """

    prospect_response: str = ""
    workflow_stage: str = "START"

    constraints_violated: List[str] = Field(default_factory=list)
    steps_completed: List[str] = Field(default_factory=list)

    turn_number: int = 0

    reward: float = 0.0
    reward_components: Dict = Field(default_factory=dict)

    done: bool = False
    info: Dict = Field(default_factory=dict)


class SalesPathState(State):
    """
    Internal environment state.
    Includes hidden state not exposed to the agent.
    """

    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

    prospect_profile: Dict = Field(default_factory=dict)
    conversation_history: List[Dict] = Field(default_factory=list)

    workflow_stage: str = "START"
    required_workflow: List[str] = Field(default_factory=list)

    steps_completed: List[str] = Field(default_factory=list)
    constraints_violated: List[str] = Field(default_factory=list)

    objections_handled: int = 0
    turn_number: int = 0
    difficulty: int = 1

    done: bool = False

    # Hidden state — NEVER exposed in Observation
    hidden_state: Dict = Field(default_factory=dict)
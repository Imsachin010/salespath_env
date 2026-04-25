# salespath_env/client.py

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import (
    SalesPathAction,
    SalesPathObservation,
    SalesPathState,
)


class SalesPathEnv(EnvClient[SalesPathAction, SalesPathObservation, SalesPathState]):

    # ------------------------------------------------------------------ #
    # Abstract method implementations required by EnvClient               #
    # ------------------------------------------------------------------ #

    def _step_payload(self, action: SalesPathAction) -> Dict[str, Any]:
        """Serialise action → JSON dict for the WebSocket server.
        WSStepMessage.data IS the action dict directly (no wrapper key).
        """
        return action.model_dump(exclude={"metadata"})

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[SalesPathObservation]:
        """Deserialise server JSON → StepResult[SalesPathObservation]."""
        # Server may nest obs under an 'observation' key
        obs_data = payload.get("observation", payload)
        obs = SalesPathObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward", obs.reward),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> SalesPathState:
        """Deserialise server JSON → SalesPathState."""
        state_data = payload.get("state", payload)
        return SalesPathState(**state_data)

    # ------------------------------------------------------------------ #
    # Convenience wrappers that return the unwrapped observation directly  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _with_step_fields(
        result: StepResult[SalesPathObservation],
    ) -> SalesPathObservation:
        """
        Keep observation fields in sync with StepResult wrapper fields.
        Some server payloads provide reward/done only at top-level.
        """
        return result.observation.model_copy(
            update={
                "reward": result.reward,
                "done": result.done,
            }
        )

    async def reset(
        self,
        difficulty: int = 1,
    ) -> SalesPathObservation:
        result = await super().reset(difficulty=difficulty)
        return self._with_step_fields(result)

    async def step(
        self,
        action_type: str,
        content: str,
        target: str = "",
    ) -> SalesPathObservation:
        action = SalesPathAction(
            action_type=action_type,
            content=content,
            target=target,
        )
        result = await super().step(action)
        return self._with_step_fields(result)
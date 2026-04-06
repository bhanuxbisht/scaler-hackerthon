from __future__ import annotations

from typing import Any

from models import SupportTriageState, TriageAction, TriageObservation
from openenv_compat import EnvClient, StepResult


class SupportTriageEnv(EnvClient[TriageAction, TriageObservation, SupportTriageState]):
    """Typed OpenEnv client for the support triage environment."""

    def _step_payload(self, action: TriageAction) -> dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[TriageObservation]:
        obs_payload = payload.get("observation", payload)
        observation = TriageObservation.model_validate(obs_payload)
        reward = payload.get("reward", observation.reward)
        done = payload.get("done", observation.done)
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
        )

    def _parse_state(self, payload: dict[str, Any]) -> SupportTriageState:
        return SupportTriageState.model_validate(payload)

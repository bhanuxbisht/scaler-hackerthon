from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, Protocol, TypeVar

from fastapi import FastAPI
from pydantic import BaseModel, Field

try:
    from openenv.core.client_types import StepResult  # type: ignore
    from openenv.core.env_client import EnvClient  # type: ignore
    from openenv.core.env_server.interfaces import Environment  # type: ignore
    from openenv.core.env_server.types import Action, Observation, State  # type: ignore

    try:
        from openenv.core.env_server.http_server import create_app  # type: ignore
    except ImportError:
        from openenv.core.env_server import create_app  # type: ignore

    OPENENV_AVAILABLE = True
except ImportError:
    OPENENV_AVAILABLE = False

    class Action(BaseModel):
        """Fallback action model when openenv-core is unavailable."""

    class Observation(BaseModel):
        """Fallback observation model."""

        done: bool = False
        reward: float = 0.0
        metadata: dict[str, Any] = Field(default_factory=dict)

    class State(BaseModel):
        """Fallback state model."""

        episode_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        step_count: int = 0

    class Environment(Protocol):
        def reset(self, **kwargs: Any) -> Observation: ...
        def step(self, action: Action) -> Observation: ...

        @property
        def state(self) -> State: ...

    ObsT = TypeVar("ObsT")

    @dataclass
    class StepResult(Generic[ObsT]):
        observation: ObsT
        reward: Optional[float]
        done: bool

    ActionT = TypeVar("ActionT")
    StateT = TypeVar("StateT")

    class EnvClient(Generic[ActionT, ObsT, StateT]):
        """Lightweight fallback client shell."""

        def __init__(self, base_url: str):
            self.base_url = base_url

        def _step_payload(self, action: ActionT) -> dict[str, Any]:
            raise NotImplementedError

        def _parse_result(self, payload: dict[str, Any]) -> StepResult[ObsT]:
            raise NotImplementedError

        def _parse_state(self, payload: dict[str, Any]) -> StateT:
            raise NotImplementedError

    def create_app(
        env_factory: Callable[[], Environment] | type[Environment],
        action_model: type[Action],
        observation_model: type[Observation],
        env_name: Optional[str] = None,
        **_: Any,
    ) -> FastAPI:
        """Fallback FastAPI app that mirrors reset/step/state endpoints."""

        app = FastAPI(title=env_name or "openenv-fallback")

        env = env_factory() if callable(env_factory) and not isinstance(env_factory, type) else env_factory()  # type: ignore[misc]

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.get("/metadata")
        def metadata() -> dict[str, Any]:
            return {
                "name": env_name or "fallback_env",
                "openenv_available": OPENENV_AVAILABLE,
            }

        @app.get("/schema")
        def schema() -> dict[str, Any]:
            return {
                "action": action_model.model_json_schema(),
                "observation": observation_model.model_json_schema(),
            }

        @app.post("/reset")
        def reset(payload: dict[str, Any] | None = None) -> dict[str, Any]:
            obs = env.reset(**(payload or {}))
            obs_dict = obs.model_dump()
            return {
                "observation": obs_dict,
                "reward": obs_dict.get("reward", 0.0),
                "done": obs_dict.get("done", False),
                "info": obs_dict.get("metadata", {}),
            }

        @app.post("/step")
        def step(payload: dict[str, Any]) -> dict[str, Any]:
            action = action_model.model_validate(payload)
            obs = env.step(action)
            obs_dict = obs.model_dump()
            return {
                "observation": obs_dict,
                "reward": obs_dict.get("reward", 0.0),
                "done": obs_dict.get("done", False),
                "info": obs_dict.get("metadata", {}),
            }

        @app.get("/state")
        def state() -> dict[str, Any]:
            return env.state.model_dump()

        return app

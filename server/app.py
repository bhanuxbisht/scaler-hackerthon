from __future__ import annotations

import uvicorn

from models import TriageAction, TriageObservation
from openenv_compat import create_app
from server.triage_environment import SupportTriageEnvironment

app = create_app(
    SupportTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="support_triage_env",
)


def main() -> None:
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from pathlib import Path

from inference import RuleAgent, run_episode
from models import TriageAction
from server.triage_environment import SupportTriageEnvironment
from tasks import list_tasks


def _assert_strict_score(value: float, label: str) -> None:
    assert 0.0 < value < 1.0, f"Score out of strict range (0,1) for {label}: {value}"


def main() -> None:
    task_specs = list_tasks()
    assert len(task_specs) >= 3, "At least 3 tasks are required."

    checks: list[dict[str, object]] = []

    for task in task_specs:
        env = SupportTriageEnvironment(default_task_id=task.task_id)
        reset_obs = env.reset(task_id=task.task_id)
        _assert_strict_score(reset_obs.progress_score, f"{task.task_id} reset progress_score")
        _assert_strict_score(reset_obs.grader_score, f"{task.task_id} reset grader_score")

        probe_obs = env.step(TriageAction(action_type="list_tickets"))
        _assert_strict_score(probe_obs.progress_score, f"{task.task_id} probe progress_score")
        _assert_strict_score(probe_obs.grader_score, f"{task.task_id} probe grader_score")

        episode = run_episode(task.task_id, RuleAgent(), max_steps=task.step_limit)
        score = episode["score"]
        _assert_strict_score(score, task.task_id)
        checks.append(
            {
                "task_id": task.task_id,
                "episode_score": score,
                "steps": episode["steps"],
                "invalid_actions": episode["invalid_actions"],
            }
        )

    output = {"checks": checks, "status": "ok"}
    out_path = Path("outputs/evals/local_validation.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()

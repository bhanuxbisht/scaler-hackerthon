from __future__ import annotations

import json
from pathlib import Path

from inference import RuleAgent, run_episode
from tasks import list_tasks


def main() -> None:
    task_specs = list_tasks()
    assert len(task_specs) >= 3, "At least 3 tasks are required."

    checks: list[dict[str, object]] = []

    for task in task_specs:
        episode = run_episode(task.task_id, RuleAgent(), max_steps=task.step_limit)
        score = episode["score"]
        assert 0.0 < score < 1.0, f"Score out of strict range (0,1) for {task.task_id}: {score}"
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

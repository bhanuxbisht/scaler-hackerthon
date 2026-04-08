from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from models import QueueName, ResolutionType, TriageAction, TriageObservation
from server.triage_environment import SupportTriageEnvironment
from tasks import list_tasks

DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL_NAME = "gpt-4.1-mini"


SYSTEM_PROMPT = """You are an autonomous support operations agent.
Return exactly one JSON object for the next action.
Do not explain.

Allowed schema:
{
  "action_type": "list_tickets|open_ticket|classify|draft_reply|resolve|advance_time|finish",
  "ticket_id": "optional ticket id",
  "priority": "low|medium|high|critical",
  "queue": "account_access|billing|technical|safety",
  "escalate": true or false,
  "tags": ["optional", "tags"],
  "reply_subject": "optional",
  "reply_body": "optional",
  "resolution": "answer_and_close|approve_refund|decline_refund|escalate_to_specialist|escalate_to_engineering",
  "minutes": 0
}
"""


def _strict_unit_interval(value: float) -> float:
    return max(0.0001, min(0.9999, value))


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _priority_from_text(text: str) -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["takeover", "unauthorized", "fraud", "all tenants", "down"]):
        return "critical"
    if any(token in lowered for token in ["invoice", "overcharge", "urgent", "board report", "etl"]):
        return "high"
    if any(token in lowered for token in ["password", "reset", "login", "refund"]):
        return "medium"
    return "low"


def _queue_from_text(text: str) -> QueueName:
    lowered = text.lower()
    if any(token in lowered for token in ["fraud", "takeover", "unauthorized", "security"]):
        return QueueName.safety
    if any(token in lowered for token in ["invoice", "refund", "billing", "charged"]):
        return QueueName.billing
    if any(token in lowered for token in ["api", "500", "outage", "timezone", "export", "etl"]):
        return QueueName.technical
    return QueueName.account_access


def _escalate_from_text(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in ["critical", "fraud", "takeover", "500", "outage", "enterprise"])


def _resolution_from_text(text: str, queue: QueueName) -> ResolutionType:
    lowered = text.lower()
    if queue == QueueName.safety:
        return ResolutionType.escalate_to_specialist
    if queue == QueueName.technical:
        return ResolutionType.escalate_to_engineering
    if queue == QueueName.billing:
        if any(token in lowered for token in ["overcharge", "duplicated seats", "invoice"]):
            return ResolutionType.approve_refund
        return ResolutionType.decline_refund
    return ResolutionType.answer_and_close


def _tags_from_text(text: str, queue: QueueName) -> list[str]:
    lowered = text.lower()
    tags: list[str] = []
    if "mfa" in lowered or "password" in lowered:
        tags.extend(["login", "mfa", "password-reset"])
    if "invoice" in lowered or "refund" in lowered or "overcharge" in lowered:
        tags.extend(["invoice", "refund", "billing"])
    if "api" in lowered or "500" in lowered or "outage" in lowered:
        tags.extend(["incident", "api", "outage"])
    if "fraud" in lowered or "unauthorized" in lowered or "takeover" in lowered:
        tags.extend(["fraud", "account-takeover", "security"])
    if "timezone" in lowered or "export" in lowered:
        tags.extend(["timezone", "export", "etl"])
    if not tags:
        if queue == QueueName.account_access:
            tags = ["login", "access"]
        elif queue == QueueName.billing:
            tags = ["billing"]
        elif queue == QueueName.technical:
            tags = ["technical"]
        else:
            tags = ["safety"]
    return sorted(set(tags))


def _draft_for(queue: QueueName, resolution: ResolutionType, subject: str) -> tuple[str, str]:
    if queue == QueueName.account_access:
        body = (
            "Thanks for reporting this. We sent a reset link and started identity verification. "
            "Once verified, access is restored within 24 hours."
        )
    elif queue == QueueName.billing and resolution == ResolutionType.approve_refund:
        body = (
            "Thanks for the invoice details. We approved a refund and logged the invoice correction. "
            "Finance will complete processing within 48 hours."
        )
    elif queue == QueueName.billing and resolution == ResolutionType.decline_refund:
        body = (
            "Thanks for your message. Per policy, we cannot refund after full digital delivery. "
            "We can offer account credit instead."
        )
    elif queue == QueueName.technical:
        body = (
            "We opened an incident and escalated to engineering. "
            "Please monitor the status page for updates. "
            "As a workaround, export in UTC until the patch ships."
        )
    else:
        body = (
            "We are helping you secure your account now. "
            "Please reset password immediately; our fraud team is engaged."
        )
    return f"Re: {subject}", body


@dataclass
class RuleAgent:
    def next_action(self, obs: TriageObservation) -> TriageAction:
        unresolved = [ticket for ticket in obs.tickets if not ticket.resolved]
        if not unresolved:
            return TriageAction(action_type="finish")

        for ticket in unresolved:
            if not ticket.opened:
                return TriageAction(action_type="open_ticket", ticket_id=ticket.ticket_id)

        for ticket in unresolved:
            if ticket.priority is None or ticket.queue is None or ticket.escalate is None:
                text = f"{ticket.subject} {ticket.body} {ticket.customer_tier}"
                queue = _queue_from_text(text)
                return TriageAction(
                    action_type="classify",
                    ticket_id=ticket.ticket_id,
                    priority=_priority_from_text(text),
                    queue=queue,
                    escalate=_escalate_from_text(text),
                    tags=_tags_from_text(text, queue),
                )

        for ticket in unresolved:
            if not ticket.draft_body:
                queue = ticket.queue or _queue_from_text(f"{ticket.subject} {ticket.body}")
                resolution = _resolution_from_text(f"{ticket.subject} {ticket.body}", queue)
                subject, body = _draft_for(queue, resolution, ticket.subject)
                return TriageAction(
                    action_type="draft_reply",
                    ticket_id=ticket.ticket_id,
                    reply_subject=subject,
                    reply_body=body,
                )

        for ticket in unresolved:
            if not ticket.resolved:
                queue = ticket.queue or _queue_from_text(f"{ticket.subject} {ticket.body}")
                resolution = _resolution_from_text(f"{ticket.subject} {ticket.body}", queue)
                return TriageAction(
                    action_type="resolve",
                    ticket_id=ticket.ticket_id,
                    resolution=resolution,
                )

        return TriageAction(action_type="finish")


class OpenAIAgent:
    def __init__(self, model_name: str, api_key: str, api_base_url: str | None, seed: int):
        self.model_name = model_name
        self.seed = seed
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_base_url or None,
            timeout=60.0,
            max_retries=2,
        )
        self.fallback = RuleAgent()
        self.max_attempts = int(os.getenv("OPENAI_STEP_MAX_ATTEMPTS", "4"))
        self.initial_backoff_s = float(os.getenv("OPENAI_STEP_BACKOFF_S", "2.0"))

    def next_action(self, obs: TriageObservation) -> TriageAction:
        serialized_obs = obs.model_dump(mode="json")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Decide the next single action for this ticket triage state:\n"
                    + json.dumps(serialized_obs, ensure_ascii=True)
                ),
            },
        ]
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    temperature=0,
                    messages=messages,
                )
                message = response.choices[0].message.content or ""
                payload = _extract_json(message)
                if payload is None:
                    return self.fallback.next_action(obs)
                return TriageAction.model_validate(payload)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_attempts:
                    sleep_s = self.initial_backoff_s * (2 ** (attempt - 1))
                    time.sleep(min(sleep_s, 20.0))
                else:
                    break

        if last_error is not None:
            print(
                f"[warn] OpenAI provider call failed after {self.max_attempts} attempts: {last_error}. "
                "Falling back to rule action for this step.",
                file=sys.stderr,
                flush=True,
            )
        return self.fallback.next_action(obs)


def run_episode(task_id: str, agent: RuleAgent | OpenAIAgent, max_steps: int) -> dict[str, Any]:
    print(f"[START] task={task_id}", flush=True)
    env = SupportTriageEnvironment(default_task_id=task_id)
    obs = env.reset(task_id=task_id)
    steps = 0
    trajectory: list[dict[str, Any]] = []

    while not obs.done and steps < max_steps:
        action = agent.next_action(obs)
        obs = env.step(action)
        current_step = steps + 1
        print(
            "[STEP] "
            f"task={task_id} "
            f"step={current_step} "
            f"action={action.action_type} "
            f"ticket={action.ticket_id or '-'} "
            f"reward={obs.reward:.4f} "
            f"done={obs.done}",
            flush=True,
        )
        trajectory.append(
            {
                "step": current_step,
                "action": action.model_dump(mode="json"),
                "reward": obs.reward,
                "done": obs.done,
                "grader_score": round(_strict_unit_interval(float(obs.grader_score)), 4),
            }
        )
        steps += 1

    if not obs.done:
        obs = env.step(TriageAction(action_type="finish"))
        print(
            "[STEP] "
            f"task={task_id} "
            f"step={steps + 1} "
            "action=finish "
            "ticket=- "
            f"reward={obs.reward:.4f} "
            f"done={obs.done}",
            flush=True,
        )
        trajectory.append(
            {
                "step": steps + 1,
                "action": {"action_type": "finish"},
                "reward": obs.reward,
                "done": obs.done,
                "grader_score": round(_strict_unit_interval(float(obs.grader_score)), 4),
            }
        )

    final_state = env.state.model_dump(mode="json")
    final_score = round(_strict_unit_interval(float(obs.grader_score)), 4)
    print(
        "[END] "
        f"task={task_id} "
        f"score={final_score:.4f} "
        f"steps={int(final_state['step_count'])} "
        f"invalid_actions={int(final_state['invalid_actions'])}",
        flush=True,
    )
    return {
        "task_id": task_id,
        "difficulty": obs.difficulty,
        "score": final_score,
        "total_reward": float(final_state["total_reward"]),
        "invalid_actions": int(final_state["invalid_actions"]),
        "steps": int(final_state["step_count"]),
        "trajectory": trajectory,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline inference for support triage OpenEnv.")
    parser.add_argument(
        "--agent",
        choices=["openai", "rule"],
        default="openai",
        help="Agent backend. Use rule mode for offline deterministic smoke tests.",
    )
    parser.add_argument("--max-steps", type=int, default=24, help="Maximum steps per task.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evals/baseline_scores.json"),
        help="Where to write JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.agent == "openai":
        api_base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
        model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
        hf_token = os.getenv("HF_TOKEN")
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY (or HF_TOKEN fallback) is required.")
        print(
            "[START] "
            f"mode=openai model={model_name} "
            f"api_base_url={api_base_url} "
            f"hf_token_present={bool(hf_token)}",
            flush=True,
        )
        agent: RuleAgent | OpenAIAgent = OpenAIAgent(
            model_name=model_name,
            api_key=api_key,
            api_base_url=api_base_url,
            seed=args.seed,
        )
    else:
        print("[START] mode=rule", flush=True)
        agent = RuleAgent()

    task_specs = list_tasks()
    results: list[dict[str, Any]] = []
    for task in task_specs:
        episode_result = run_episode(task.task_id, agent=agent, max_steps=args.max_steps)
        results.append(episode_result)

    avg_score = round(sum(result["score"] for result in results) / len(results), 4)
    output = {
        "agent_mode": args.agent,
        "model_name": os.getenv("MODEL_NAME", "rule-based"),
        "api_base_url": os.getenv("API_BASE_URL", ""),
        "scores": results,
        "average_score": avg_score,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("[END] summary=baseline_results", flush=True)
    for result in results:
        print(
            "[STEP] "
            f"task={result['task_id']} "
            f"difficulty={result['difficulty']} "
            f"score={result['score']:.4f} "
            f"steps={result['steps']} "
            f"invalid={result['invalid_actions']}",
            flush=True,
        )
    print(f"[END] average_score={avg_score:.4f}", flush=True)
    print(f"[END] saved={args.output}", flush=True)


if __name__ == "__main__":
    main()

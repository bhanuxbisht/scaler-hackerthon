from __future__ import annotations

import json
from uuid import uuid4

from graders import grade_task, partial_progress
from models import (
    ResolutionType,
    SupportTriageState,
    TicketSnapshot,
    TriageAction,
    TriageObservation,
    TriageReward,
)
from openenv_compat import Environment
from tasks import TaskSpec, get_task, list_tasks


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _strict_unit_interval(value: float) -> float:
    return _clamp(value, 0.0001, 0.9999)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _contains(text: str, phrase: str) -> bool:
    return _normalize(phrase) in _normalize(text)


class SupportTriageEnvironment(Environment):
    """Customer-support ticket triage environment with policy-aware grading."""

    def __init__(self, default_task_id: str = "support_triage_easy"):
        self._default_task_id = default_task_id
        self._task: TaskSpec = get_task(default_task_id)
        self._state: SupportTriageState = self._new_state(self._task)
        self._done = False

    def _new_state(self, task: TaskSpec) -> SupportTriageState:
        return SupportTriageState(
            episode_id=str(uuid4()),
            step_count=0,
            task_id=task.task_id,
            task_title=task.title,
            difficulty=task.difficulty,
            current_time_min=0,
            step_limit=task.step_limit,
            invalid_actions=0,
            total_reward=0.0,
            task_complete=False,
            active_ticket_id=None,
            tickets={
                ticket.ticket_id: TicketSnapshot(
                    ticket_id=ticket.ticket_id,
                    customer_tier=ticket.customer_tier,
                    subject=ticket.subject,
                    body=ticket.body,
                )
                for ticket in task.tickets
            },
            recent_actions=[],
        )

    def reset(self, task_id: str | None = None, **kwargs: object) -> TriageObservation:
        chosen_task_id = task_id or kwargs.get("task_id") or self._default_task_id
        self._task = get_task(str(chosen_task_id))
        self._state = self._new_state(self._task)
        self._done = False
        initial_progress = partial_progress(self._task, self._state)
        return self._build_observation(
            message=(
                "Episode reset. Start by listing tickets, then open/classify/draft/resolve "
                "each ticket according to urgency and policy."
            ),
            reward=0.0,
            done=False,
            breakdown=TriageReward(),
            progress=initial_progress,
            grader_score=initial_progress,
        )

    def step(self, action: TriageAction) -> TriageObservation:
        if self._done:
            return self._build_observation(
                message="Episode already complete. Call reset() to start a new episode.",
                reward=-0.05,
                done=True,
                breakdown=TriageReward(
                    progress_delta=0.0,
                    quality_bonus=0.0,
                    penalty=0.05,
                    total=-0.05,
                ),
                progress=partial_progress(self._task, self._state),
                grader_score=grade_task(self._task, self._state),
            )

        self._state.step_count += 1
        prev_progress = partial_progress(self._task, self._state)
        quality_bonus = 0.0
        penalty = 0.0
        invalid = False
        force_done = False
        message = ""

        action_sig = self._action_signature(action)
        self._state.recent_actions.append(action_sig)
        if len(self._state.recent_actions) > 12:
            self._state.recent_actions = self._state.recent_actions[-12:]
        if len(self._state.recent_actions) >= 3 and len(set(self._state.recent_actions[-3:])) == 1:
            penalty += 0.08
            message += "Repeated identical action pattern detected. "

        if action.action_type == "list_tickets":
            unresolved = [tid for tid, t in self._state.tickets.items() if not t.resolved]
            message += f"Unresolved tickets: {', '.join(unresolved) if unresolved else 'none'}."

        elif action.action_type == "open_ticket":
            ticket = self._get_runtime_ticket(action.ticket_id)
            if ticket is None:
                invalid = True
                message += "Invalid ticket_id for open_ticket."
            else:
                first_open = not ticket.opened
                ticket.opened = True
                self._state.active_ticket_id = ticket.ticket_id
                if first_open:
                    quality_bonus += 0.02
                message += f"Opened ticket {ticket.ticket_id}."

        elif action.action_type == "classify":
            runtime = self._get_runtime_ticket(action.ticket_id)
            spec = self._get_spec_ticket(action.ticket_id)
            if runtime is None or spec is None:
                invalid = True
                message += "Invalid ticket_id for classify."
            elif action.priority is None or action.queue is None or action.escalate is None:
                invalid = True
                message += "classify requires priority, queue, and escalate."
            else:
                runtime.priority = action.priority
                runtime.queue = action.queue
                runtime.escalate = action.escalate
                runtime.tags = sorted(set(tag.lower() for tag in action.tags))
                if runtime.priority == spec.expected_priority:
                    quality_bonus += 0.03
                else:
                    penalty += 0.03
                if runtime.queue == spec.expected_queue:
                    quality_bonus += 0.03
                else:
                    penalty += 0.03
                if runtime.escalate == spec.expected_escalate:
                    quality_bonus += 0.03
                else:
                    penalty += 0.03
                if not runtime.opened:
                    penalty += 0.02
                message += f"Classified ticket {runtime.ticket_id}."

        elif action.action_type == "draft_reply":
            runtime = self._get_runtime_ticket(action.ticket_id)
            spec = self._get_spec_ticket(action.ticket_id)
            if runtime is None or spec is None:
                invalid = True
                message += "Invalid ticket_id for draft_reply."
            elif not action.reply_body:
                invalid = True
                message += "draft_reply requires reply_body."
            else:
                runtime.draft_subject = action.reply_subject or f"Re: {runtime.subject}"
                runtime.draft_body = action.reply_body
                combined = f"{runtime.draft_subject} {runtime.draft_body}"
                if spec.required_reply_phrases:
                    hits = sum(1 for phrase in spec.required_reply_phrases if _contains(combined, phrase))
                    coverage = hits / len(spec.required_reply_phrases)
                else:
                    coverage = 1.0
                quality_bonus += 0.08 * coverage
                forbidden_hits = sum(
                    1 for phrase in spec.forbidden_reply_phrases if _contains(combined, phrase)
                )
                penalty += min(0.20, forbidden_hits * 0.08)
                if not runtime.opened:
                    penalty += 0.02
                message += f"Drafted reply for {runtime.ticket_id} (coverage={coverage:.2f})."

        elif action.action_type == "resolve":
            runtime = self._get_runtime_ticket(action.ticket_id)
            spec = self._get_spec_ticket(action.ticket_id)
            if runtime is None or spec is None:
                invalid = True
                message += "Invalid ticket_id for resolve."
            elif action.resolution is None:
                invalid = True
                message += "resolve requires resolution."
            elif runtime.resolved:
                invalid = True
                message += f"Ticket {runtime.ticket_id} is already resolved."
            else:
                runtime.resolution = action.resolution
                runtime.resolved = True
                runtime.resolved_at_minute = self._state.current_time_min
                if runtime.resolution == spec.expected_resolution:
                    quality_bonus += 0.10
                else:
                    penalty += 0.08
                if runtime.priority is None or runtime.queue is None or runtime.escalate is None:
                    penalty += 0.05
                if not runtime.draft_body:
                    penalty += 0.05
                if runtime.resolved_at_minute <= spec.sla_minutes:
                    quality_bonus += 0.05
                else:
                    penalty += 0.06
                message += f"Resolved ticket {runtime.ticket_id} with {runtime.resolution.value}."

        elif action.action_type == "advance_time":
            if action.minutes <= 0:
                invalid = True
                message += "advance_time requires minutes > 0."
            else:
                self._state.current_time_min += action.minutes
                penalty += min(0.08, action.minutes / 300.0)
                message += f"Advanced simulated time by {action.minutes} minutes."

        elif action.action_type == "finish":
            force_done = True
            unresolved = [tid for tid, t in self._state.tickets.items() if not t.resolved]
            if unresolved:
                penalty += 0.05
                message += f"Finished early with unresolved tickets: {', '.join(unresolved)}."
            else:
                quality_bonus += 0.03
                message += "Finished episode."

        if invalid:
            self._state.invalid_actions += 1
            penalty += 0.12

        new_progress = partial_progress(self._task, self._state)
        progress_delta = new_progress - prev_progress
        reward = 1.2 * progress_delta + quality_bonus - penalty - 0.01
        reward = round(_clamp(reward, -1.0, 1.0), 4)
        self._state.total_reward = round(self._state.total_reward + reward, 4)

        step_limit_reached = self._state.step_count >= self._state.step_limit
        all_resolved = all(ticket.resolved for ticket in self._state.tickets.values())
        done = force_done or all_resolved or step_limit_reached
        self._done = done
        grader_score = grade_task(self._task, self._state) if done else new_progress
        self._state.task_complete = done and grader_score >= 0.8

        if step_limit_reached and not force_done and not all_resolved:
            message += " Step limit reached."

        breakdown = TriageReward(
            progress_delta=round(progress_delta, 4),
            quality_bonus=round(quality_bonus, 4),
            penalty=round(penalty, 4),
            total=reward,
        )
        return self._build_observation(
            message=message.strip(),
            reward=reward,
            done=done,
            breakdown=breakdown,
            progress=new_progress,
            grader_score=grader_score,
        )

    def _action_signature(self, action: TriageAction) -> str:
        payload = action.model_dump(exclude_none=True, exclude_defaults=True, mode="json")
        return json.dumps(payload, sort_keys=True)

    def _get_runtime_ticket(self, ticket_id: str | None) -> TicketSnapshot | None:
        if not ticket_id:
            return None
        return self._state.tickets.get(ticket_id)

    def _get_spec_ticket(self, ticket_id: str | None):
        if not ticket_id:
            return None
        for ticket in self._task.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _build_observation(
        self,
        *,
        message: str,
        reward: float,
        done: bool,
        breakdown: TriageReward,
        progress: float,
        grader_score: float,
    ) -> TriageObservation:
        strict_progress = round(_strict_unit_interval(progress), 4)
        strict_grader = round(_strict_unit_interval(grader_score), 4)
        remaining = sorted(
            ticket_id for ticket_id, ticket in self._state.tickets.items() if not ticket.resolved
        )
        metadata = {
            "task_id": self._task.task_id,
            "task_title": self._task.title,
            "difficulty": self._task.difficulty,
            "grader_score": strict_grader,
            "progress_score": strict_progress,
            "invalid_actions": self._state.invalid_actions,
            "remaining_tickets": remaining,
            "step_count": self._state.step_count,
        }
        return TriageObservation(
            message=message,
            objective=self._task.objective,
            task_id=self._task.task_id,
            difficulty=self._task.difficulty,
            current_time_min=self._state.current_time_min,
            active_ticket_id=self._state.active_ticket_id,
            remaining_ticket_ids=remaining,
            tickets=list(self._state.tickets.values()),
            reward_breakdown=breakdown,
            progress_score=strict_progress,
            grader_score=strict_grader,
            done=done,
            reward=reward,
            metadata=metadata,
        )

    @property
    def state(self) -> SupportTriageState:
        return self._state

    @staticmethod
    def available_tasks() -> list[dict[str, str]]:
        return [
            {"task_id": task.task_id, "difficulty": task.difficulty, "title": task.title}
            for task in list_tasks()
        ]

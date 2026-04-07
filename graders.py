from __future__ import annotations

from typing import Callable

from models import SupportTriageState
from tasks import TaskSpec, TicketSpec, get_task


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _strict_unit_interval(value: float) -> float:
    # Hackathon validator expects strict bounds: 0.0 < score < 1.0.
    return _clamp(value, 0.0001, 0.9999)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _phrase_coverage(text: str, required_phrases: list[str]) -> float:
    if not required_phrases:
        return 1.0
    normalized = _normalize(text)
    hit_count = sum(1 for phrase in required_phrases if _normalize(phrase) in normalized)
    return hit_count / len(required_phrases)


def _forbidden_hits(text: str, forbidden_phrases: list[str]) -> int:
    normalized = _normalize(text)
    return sum(1 for phrase in forbidden_phrases if _normalize(phrase) in normalized)


def _grade_ticket(ticket: TicketSpec, state: SupportTriageState) -> float:
    runtime = state.tickets.get(ticket.ticket_id)
    if runtime is None:
        return 0.0

    cls_correct = 0.0
    cls_total = 3.0
    cls_correct += 1.0 if runtime.priority == ticket.expected_priority else 0.0
    cls_correct += 1.0 if runtime.queue == ticket.expected_queue else 0.0
    cls_correct += 1.0 if runtime.escalate == ticket.expected_escalate else 0.0
    classification_score = cls_correct / cls_total

    tags_required = set(tag.lower() for tag in ticket.expected_tags)
    tags_observed = set(tag.lower() for tag in runtime.tags)
    if tags_required:
        tags_score = len(tags_required.intersection(tags_observed)) / len(tags_required)
    else:
        tags_score = 1.0

    reply_text = f"{runtime.draft_subject or ''} {runtime.draft_body or ''}"
    coverage = _phrase_coverage(reply_text, ticket.required_reply_phrases)
    forbidden_count = _forbidden_hits(reply_text, ticket.forbidden_reply_phrases)
    draft_score = _clamp(coverage - 0.25 * forbidden_count)

    resolution_score = 1.0 if runtime.resolution == ticket.expected_resolution else 0.0
    if runtime.resolved and runtime.resolution is not None and runtime.resolution != ticket.expected_resolution:
        resolution_score = 0.15

    sla_score = 0.0
    if runtime.resolved and runtime.resolved_at_minute is not None:
        sla_score = 1.0 if runtime.resolved_at_minute <= ticket.sla_minutes else 0.0

    weighted = (
        0.35 * (0.75 * classification_score + 0.25 * tags_score)
        + 0.25 * draft_score
        + 0.30 * resolution_score
        + 0.10 * sla_score
    )
    return _clamp(weighted)


def _grade_order(task: TaskSpec, state: SupportTriageState) -> float:
    if len(task.resolve_order) <= 1:
        return 1.0
    pairs = list(zip(task.resolve_order[:-1], task.resolve_order[1:]))
    valid_pairs = 0
    for first, second in pairs:
        first_state = state.tickets.get(first)
        second_state = state.tickets.get(second)
        if (
            first_state is not None
            and second_state is not None
            and first_state.resolved_at_minute is not None
            and second_state.resolved_at_minute is not None
            and first_state.resolved_at_minute <= second_state.resolved_at_minute
        ):
            valid_pairs += 1
    return valid_pairs / len(pairs)


def grade_task(task: TaskSpec, state: SupportTriageState) -> float:
    ticket_scores = [_grade_ticket(ticket, state) for ticket in task.tickets]
    ticket_average = sum(ticket_scores) / max(len(ticket_scores), 1)
    order_score = _grade_order(task, state)
    score = 0.9 * ticket_average + 0.1 * order_score
    score -= min(0.20, 0.02 * state.invalid_actions)
    return round(_strict_unit_interval(score), 4)


def partial_progress(task: TaskSpec, state: SupportTriageState) -> float:
    totals: list[float] = []
    for ticket in task.tickets:
        runtime = state.tickets.get(ticket.ticket_id)
        if runtime is None:
            totals.append(0.0)
            continue

        opened_score = 0.15 if runtime.opened else 0.0
        cls_score = 0.0
        cls_score += 1.0 if runtime.priority == ticket.expected_priority else 0.0
        cls_score += 1.0 if runtime.queue == ticket.expected_queue else 0.0
        cls_score += 1.0 if runtime.escalate == ticket.expected_escalate else 0.0
        cls_score = (cls_score / 3.0) * 0.30

        required_tags = set(tag.lower() for tag in ticket.expected_tags)
        observed_tags = set(tag.lower() for tag in runtime.tags)
        if required_tags:
            tags_score = (len(required_tags.intersection(observed_tags)) / len(required_tags)) * 0.10
        else:
            tags_score = 0.10

        reply_text = f"{runtime.draft_subject or ''} {runtime.draft_body or ''}"
        draft_score = _phrase_coverage(reply_text, ticket.required_reply_phrases) * 0.20
        draft_score -= min(0.20, 0.05 * _forbidden_hits(reply_text, ticket.forbidden_reply_phrases))
        draft_score = _clamp(draft_score, 0.0, 0.20)

        resolution_score = 0.0
        if runtime.resolved:
            resolution_score = 0.25 if runtime.resolution == ticket.expected_resolution else 0.12

        totals.append(opened_score + cls_score + tags_score + draft_score + resolution_score)

    progress = sum(totals) / max(len(totals), 1)
    progress += 0.05 * _grade_order(task, state)
    progress -= min(0.10, 0.01 * state.invalid_actions)
    return round(_clamp(progress), 4)


def grade_task_by_id(task_id: str, state: SupportTriageState) -> float:
    return grade_task(get_task(task_id), state)


def grade_easy(state: SupportTriageState) -> float:
    return grade_task_by_id("support_triage_easy", state)


def grade_medium(state: SupportTriageState) -> float:
    return grade_task_by_id("support_triage_medium", state)


def grade_hard(state: SupportTriageState) -> float:
    return grade_task_by_id("support_triage_hard", state)


TASK_GRADERS: dict[str, Callable[[SupportTriageState], float]] = {
    "support_triage_easy": grade_easy,
    "support_triage_medium": grade_medium,
    "support_triage_hard": grade_hard,
}

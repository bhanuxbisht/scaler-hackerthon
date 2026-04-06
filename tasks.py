from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from models import Priority, QueueName, ResolutionType


class TicketSpec(BaseModel):
    ticket_id: str
    customer_tier: Literal["standard", "pro", "enterprise"]
    subject: str
    body: str
    expected_priority: Priority
    expected_queue: QueueName
    expected_escalate: bool
    expected_tags: list[str] = Field(default_factory=list)
    expected_resolution: ResolutionType
    required_reply_phrases: list[str] = Field(default_factory=list)
    forbidden_reply_phrases: list[str] = Field(default_factory=list)
    sla_minutes: int = 240


class TaskSpec(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    title: str
    objective: str
    step_limit: int
    tickets: list[TicketSpec]
    resolve_order: list[str] = Field(default_factory=list)


TASKS: dict[str, TaskSpec] = {
    "support_triage_easy": TaskSpec(
        task_id="support_triage_easy",
        difficulty="easy",
        title="Single Account Access Ticket",
        objective=(
            "Triage one account-access request. Correctly classify priority/queue, "
            "send a compliant reply, and close the ticket within SLA."
        ),
        step_limit=12,
        tickets=[
            TicketSpec(
                ticket_id="E-100",
                customer_tier="standard",
                subject="Can not reset password after changing phone",
                body=(
                    "I replaced my phone and can not receive MFA prompts. "
                    "Please help me sign in today."
                ),
                expected_priority=Priority.medium,
                expected_queue=QueueName.account_access,
                expected_escalate=False,
                expected_tags=["login", "mfa", "password-reset"],
                expected_resolution=ResolutionType.answer_and_close,
                required_reply_phrases=[
                    "reset link",
                    "identity verification",
                    "24 hours",
                ],
                forbidden_reply_phrases=["share your password"],
                sla_minutes=120,
            )
        ],
        resolve_order=["E-100"],
    ),
    "support_triage_medium": TaskSpec(
        task_id="support_triage_medium",
        difficulty="medium",
        title="Incident Plus Billing Correction",
        objective=(
            "Handle two concurrent tickets: a production outage and a billing overcharge. "
            "Resolve in priority order, use escalation where required, and keep replies policy-safe."
        ),
        step_limit=18,
        tickets=[
            TicketSpec(
                ticket_id="M-210",
                customer_tier="enterprise",
                subject="API returns 500 for all tenants",
                body=(
                    "Since 09:10 UTC, every API request fails with HTTP 500. "
                    "Our customer dashboard is down for all users."
                ),
                expected_priority=Priority.critical,
                expected_queue=QueueName.technical,
                expected_escalate=True,
                expected_tags=["incident", "api", "outage"],
                expected_resolution=ResolutionType.escalate_to_engineering,
                required_reply_phrases=[
                    "incident",
                    "engineering",
                    "status page",
                ],
                forbidden_reply_phrases=["resolved already"],
                sla_minutes=30,
            ),
            TicketSpec(
                ticket_id="M-211",
                customer_tier="enterprise",
                subject="Invoice overcharged by $1,200",
                body=(
                    "Our March invoice includes duplicated seats. Finance attached screenshots. "
                    "Please process correction quickly."
                ),
                expected_priority=Priority.high,
                expected_queue=QueueName.billing,
                expected_escalate=True,
                expected_tags=["invoice", "overcharge", "refund"],
                expected_resolution=ResolutionType.approve_refund,
                required_reply_phrases=[
                    "refund",
                    "invoice",
                    "48 hours",
                ],
                forbidden_reply_phrases=["guarantee immediate wire transfer"],
                sla_minutes=240,
            ),
        ],
        resolve_order=["M-210", "M-211"],
    ),
    "support_triage_hard": TaskSpec(
        task_id="support_triage_hard",
        difficulty="hard",
        title="Risk, Policy, and Platform Queue",
        objective=(
            "Process three mixed-criticality tickets with strict policy compliance: "
            "security takeover, policy-limited refund request, and enterprise export bug."
        ),
        step_limit=24,
        tickets=[
            TicketSpec(
                ticket_id="H-310",
                customer_tier="pro",
                subject="Possible account takeover and unauthorized charges",
                body=(
                    "I saw login alerts from another country and two unknown purchases. "
                    "Please lock my account now."
                ),
                expected_priority=Priority.critical,
                expected_queue=QueueName.safety,
                expected_escalate=True,
                expected_tags=["fraud", "account-takeover", "security"],
                expected_resolution=ResolutionType.escalate_to_specialist,
                required_reply_phrases=[
                    "secure your account",
                    "reset password",
                    "fraud team",
                ],
                forbidden_reply_phrases=["share your password"],
                sla_minutes=20,
            ),
            TicketSpec(
                ticket_id="H-311",
                customer_tier="standard",
                subject="Requesting refund after downloading full report",
                body=(
                    "I purchased the report last month, downloaded everything, "
                    "and now want a full refund."
                ),
                expected_priority=Priority.medium,
                expected_queue=QueueName.billing,
                expected_escalate=False,
                expected_tags=["refund", "policy", "billing"],
                expected_resolution=ResolutionType.decline_refund,
                required_reply_phrases=[
                    "policy",
                    "cannot refund",
                    "credit",
                ],
                forbidden_reply_phrases=["full refund approved"],
                sla_minutes=360,
            ),
            TicketSpec(
                ticket_id="H-312",
                customer_tier="enterprise",
                subject="CSV exports shifted timestamps by timezone",
                body=(
                    "Exports in CSV are shifted by +5:30 and break downstream ETL. "
                    "Need workaround before tomorrow board report."
                ),
                expected_priority=Priority.high,
                expected_queue=QueueName.technical,
                expected_escalate=True,
                expected_tags=["timezone", "export", "etl"],
                expected_resolution=ResolutionType.escalate_to_engineering,
                required_reply_phrases=[
                    "engineering",
                    "workaround",
                    "utc",
                ],
                forbidden_reply_phrases=["fixed globally"],
                sla_minutes=120,
            ),
        ],
        resolve_order=["H-310", "H-312", "H-311"],
    ),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in TASKS:
        valid = ", ".join(sorted(TASKS.keys()))
        raise ValueError(f"Unknown task_id='{task_id}'. Valid values: {valid}")
    return TASKS[task_id]


def list_tasks() -> list[TaskSpec]:
    return [TASKS["support_triage_easy"], TASKS["support_triage_medium"], TASKS["support_triage_hard"]]


def default_task_id() -> str:
    return "support_triage_easy"

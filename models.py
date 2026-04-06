from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field

from openenv_compat import Action, Observation, State


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class QueueName(str, Enum):
    account_access = "account_access"
    billing = "billing"
    technical = "technical"
    safety = "safety"


class ResolutionType(str, Enum):
    answer_and_close = "answer_and_close"
    approve_refund = "approve_refund"
    decline_refund = "decline_refund"
    escalate_to_specialist = "escalate_to_specialist"
    escalate_to_engineering = "escalate_to_engineering"


class TriageReward(BaseModel):
    progress_delta: float = 0.0
    quality_bonus: float = 0.0
    penalty: float = 0.0
    total: float = 0.0


class TicketSnapshot(BaseModel):
    ticket_id: str
    customer_tier: str
    subject: str
    body: str
    opened: bool = False
    priority: Optional[Priority] = None
    queue: Optional[QueueName] = None
    escalate: Optional[bool] = None
    tags: list[str] = Field(default_factory=list)
    draft_subject: Optional[str] = None
    draft_body: Optional[str] = None
    resolved: bool = False
    resolution: Optional[ResolutionType] = None
    resolved_at_minute: Optional[int] = None


class SupportTriageState(State):
    task_id: str = ""
    task_title: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    current_time_min: int = 0
    step_limit: int = 20
    invalid_actions: int = 0
    total_reward: float = 0.0
    task_complete: bool = False
    active_ticket_id: Optional[str] = None
    tickets: dict[str, TicketSnapshot] = Field(default_factory=dict)
    recent_actions: list[str] = Field(default_factory=list)


class TriageAction(Action):
    action_type: Literal[
        "list_tickets",
        "open_ticket",
        "classify",
        "draft_reply",
        "resolve",
        "advance_time",
        "finish",
    ] = Field(..., description="Action type to execute.")
    ticket_id: Optional[str] = Field(
        default=None, description="Target ticket id for ticket-specific actions."
    )
    priority: Optional[Priority] = None
    queue: Optional[QueueName] = None
    escalate: Optional[bool] = None
    tags: list[str] = Field(default_factory=list)
    reply_subject: Optional[str] = None
    reply_body: Optional[str] = None
    resolution: Optional[ResolutionType] = None
    minutes: int = Field(
        default=0, ge=0, le=240, description="Minutes to advance simulated clock."
    )


class TriageObservation(Observation):
    message: str = ""
    objective: str = ""
    task_id: str = ""
    difficulty: Literal["easy", "medium", "hard"] = "easy"
    current_time_min: int = 0
    active_ticket_id: Optional[str] = None
    remaining_ticket_ids: list[str] = Field(default_factory=list)
    tickets: list[TicketSnapshot] = Field(default_factory=list)
    reward_breakdown: TriageReward = Field(default_factory=TriageReward)
    progress_score: float = 0.0
    grader_score: float = 0.0
    available_actions: list[str] = Field(
        default_factory=lambda: [
            "list_tickets",
            "open_ticket",
            "classify",
            "draft_reply",
            "resolve",
            "advance_time",
            "finish",
        ]
    )

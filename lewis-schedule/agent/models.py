from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Day = Literal[
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
]
Period = Literal["morning", "afternoon"]


class Slot(BaseModel):
    day: Day
    period: Period
    activity: str | None = ""
    caregiver: str | None = ""
    time: str | None = ""


class ScheduleState(BaseModel):
    week_start: str | None = None
    caregivers: list[str] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    slots: list[Slot] = Field(default_factory=list)


class ImportStartRequest(BaseModel):
    week_start: str | None = None
    schedule: ScheduleState
    image_base64: str
    mime_type: str = "image/jpeg"


class ImportContinueRequest(BaseModel):
    thread_id: str
    user_message: str
    schedule: ScheduleState


class Question(BaseModel):
    id: str
    text: str
    choices: list[str] = Field(default_factory=list)


class PatchEntry(BaseModel):
    day: Day
    period: Period
    activity: str | None = None
    caregiver: str | None = None
    time: str | None = None


class AgentResponse(BaseModel):
    mode: Literal["questions", "proposal", "noop"]
    message: str = ""
    questions: list[Question] = Field(default_factory=list)
    patch: list[PatchEntry] = Field(default_factory=list)


class ImportApiResponse(BaseModel):
    thread_id: str
    agent: AgentResponse
    raw_text: str | None = None


class HealthResponse(BaseModel):
    ok: bool
    workspace: str
    composer_available: bool
    warm_agent_ready: bool = False


class ScheduleSaveResponse(BaseModel):
    ok: bool
    week_start: str


class ChatRequest(BaseModel):
    message: str
    schedule: ScheduleState
    thread_id: str | None = None


class ChatAgentReply(BaseModel):
    message: str = ""
    patch: list[PatchEntry] = Field(default_factory=list)


class ChatApiResponse(BaseModel):
    thread_id: str
    reply: ChatAgentReply
    raw_text: str | None = None

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from cursor_sdk import (
    Agent,
    AgentOptions,
    CloudAgentOptions,
    CloudRepository,
    SDKImage,
    UserMessage,
)

from models import AgentResponse, ChatAgentReply, ChatApiResponse, ImportApiResponse, ScheduleState


AGENT_MODEL = "auto"
SKILL_PATH = ".cursor/skills/lewis-schedule-import/SKILL.md"
DEFAULT_CLOUD_REPO = "https://github.com/kthngan/Cursor"


def monday_of_week(reference: datetime | None = None) -> str:
    ref = reference or datetime.now(UTC)
    monday = ref - timedelta(days=ref.weekday())
    return monday.date().isoformat()


def extract_json_from_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Empty agent response")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", cleaned, re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1).strip())

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        return json.loads(cleaned[start : end + 1])

    raise ValueError("Agent response did not contain JSON")


def parse_agent_response(text: str) -> AgentResponse:
    payload = extract_json_from_text(text)
    return AgentResponse.model_validate(payload)


def build_import_prompt(
    *,
    schedule: ScheduleState,
    week_start: str | None,
    skill_text: str,
    user_message: str | None = None,
    is_start: bool,
) -> str:
    schedule_json = json.dumps(schedule.model_dump(), indent=2)
    parts = [
        "You are updating Lewis's weekly schedule from a partial screenshot.",
        "Follow the skill instructions and return JSON only.",
        "Do not edit repository files. Only return the JSON response.",
        "",
        "## Skill",
        skill_text,
        "",
        f"## Week start: {week_start or monday_of_week()}",
        "## Current schedule",
        schedule_json,
    ]
    if is_start:
        parts.extend(
            [
                "",
                "## Task",
                "The user uploaded a screenshot. Read it and either ask clarifying questions",
                "or propose a minimal patch. Return JSON only.",
            ]
        )
    else:
        parts.extend(
            [
                "",
                "## User follow-up",
                user_message or "",
                "",
                "Continue the import conversation. Return JSON only.",
            ]
        )
    return "\n".join(parts)


def build_chat_prompt(
    *,
    schedule: ScheduleState,
    week_start: str | None,
    skill_text: str,
    user_message: str,
    is_start: bool,
) -> str:
    schedule_json = json.dumps(schedule.model_dump(), indent=2)
    parts = [
        "You are a helpful assistant for Lewis's weekly care schedule.",
        "Answer in a friendly, concise way. You can explain the schedule or suggest edits.",
        "Follow the schedule skill rules when proposing changes.",
        "Do not edit repository files. Only return the JSON response.",
        "",
        "## Skill",
        skill_text,
        "",
        f"## Week start: {week_start or monday_of_week()}",
        "## Current schedule",
        schedule_json,
        "",
        "## Response format",
        "Reply with JSON only (no markdown fences):",
        '{ "message": "your reply to the user", "patch": [] }',
        "Use patch only when proposing schedule changes (same shape as import proposals).",
        "Leave patch as [] when you are only chatting or asking a question.",
        "Valid days: monday..saturday. Valid periods: morning, afternoon.",
        "Optional time field is HH:MM and only when activity is set.",
    ]
    if is_start:
        parts.extend(["", "## Conversation start", "User message:", user_message])
    else:
        parts.extend(["", "## User message", user_message])
    return "\n".join(parts)


def parse_chat_response(text: str) -> ChatAgentReply:
    try:
        payload = extract_json_from_text(text)
        return ChatAgentReply.model_validate(payload)
    except Exception:
        return ChatAgentReply(message=text.strip() or "I could not parse a reply.", patch=[])


@dataclass
class ImportSession:
    thread_id: str
    agent: Any
    stack: AsyncExitStack
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ScheduleAgentService:
    def __init__(
        self,
        *,
        workspace: str,
        api_key: str,
        cloud_repo_url: str | None = None,
    ) -> None:
        self.workspace = workspace
        self.api_key = api_key
        self.cloud_repo_url = (
            cloud_repo_url
            or os.environ.get("CLOUD_REPO_URL")
            or DEFAULT_CLOUD_REPO
        ).rstrip("/")
        self._sessions: dict[str, ImportSession] = {}
        self._skill_text: str | None = None

    @property
    def composer_available(self) -> bool:
        return bool(self.api_key)

    async def startup(self) -> None:
        self._skill_text = self._read_skill_text()

    async def shutdown(self) -> None:
        for session in list(self._sessions.values()):
            await self.close_session(session.thread_id)

    def _read_skill_text(self) -> str:
        skill_file = Path(self.workspace) / SKILL_PATH
        if skill_file.is_file():
            return skill_file.read_text(encoding="utf-8")
        return "Return schedule import JSON with modes questions, proposal, or noop."

    def _cloud_options(self) -> CloudAgentOptions:
        return CloudAgentOptions(
            repos=[
                CloudRepository(
                    url=self.cloud_repo_url,
                    starting_ref="main",
                )
            ],
            auto_create_pr=False,
            skip_reviewer_request=True,
        )

    def _create_cloud_agent_sync(self) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "Cloud agent is not configured. Set CURSOR_API_KEY in lewis-schedule/agent/.env"
            )
        return Agent.create(
            model=AGENT_MODEL,
            api_key=self.api_key,
            name="Lewis Schedule",
            cloud=self._cloud_options(),
        )

    def _resume_cloud_agent_sync(self, agent_id: str) -> Any:
        if not self.api_key:
            raise RuntimeError(
                "Cloud agent is not configured. Set CURSOR_API_KEY in lewis-schedule/agent/.env"
            )
        return Agent.resume(
            agent_id,
            AgentOptions(api_key=self.api_key),
        )

    def _close_agent_sync(self, agent: Any) -> None:
        close = getattr(agent, "close", None)
        if callable(close):
            close()

    def _send_sync(
        self,
        *,
        agent: Any,
        prompt: str,
        image_base64: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> str:
        images = []
        if image_base64:
            images.append(SDKImage.from_data(image_base64, mime_type))
        run = agent.send(UserMessage(text=prompt, images=images or None))
        return run.text()

    async def _create_agent_session(self) -> ImportSession:
        agent = await asyncio.to_thread(self._create_cloud_agent_sync)
        thread_id = str(getattr(agent, "agent_id", "") or "")
        if not thread_id:
            raise RuntimeError("Cloud agent did not return an agent_id")
        stack = AsyncExitStack()
        session = ImportSession(thread_id=thread_id, agent=agent, stack=stack)
        self._sessions[thread_id] = session
        return session

    async def _get_or_resume_session(self, thread_id: str) -> ImportSession:
        existing = self._sessions.get(thread_id)
        if existing is not None:
            return existing
        agent = await asyncio.to_thread(self._resume_cloud_agent_sync, thread_id)
        stack = AsyncExitStack()
        session = ImportSession(thread_id=thread_id, agent=agent, stack=stack)
        self._sessions[thread_id] = session
        return session

    async def close_session(self, thread_id: str) -> None:
        session = self._sessions.pop(thread_id, None)
        if session is None:
            return
        await asyncio.to_thread(self._close_agent_sync, session.agent)
        await session.stack.aclose()

    async def _send_to_agent(
        self,
        *,
        session: ImportSession,
        prompt: str,
        image_base64: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> str:
        return await asyncio.to_thread(
            self._send_sync,
            agent=session.agent,
            prompt=prompt,
            image_base64=image_base64,
            mime_type=mime_type,
        )

    async def start_import(
        self,
        *,
        schedule: ScheduleState,
        week_start: str | None,
        image_base64: str,
        mime_type: str,
    ) -> ImportApiResponse:
        session = await self._create_agent_session()
        prompt = build_import_prompt(
            schedule=schedule,
            week_start=week_start,
            skill_text=self._skill_text or "",
            is_start=True,
        )
        raw_text = await self._send_to_agent(
            session=session,
            prompt=prompt,
            image_base64=image_base64,
            mime_type=mime_type,
        )
        agent = parse_agent_response(raw_text)
        return ImportApiResponse(
            thread_id=session.thread_id,
            agent=agent,
            raw_text=raw_text,
        )

    async def continue_import(
        self,
        *,
        thread_id: str,
        schedule: ScheduleState,
        week_start: str | None,
        user_message: str,
    ) -> ImportApiResponse:
        try:
            session = await self._get_or_resume_session(thread_id)
        except Exception as exc:
            raise KeyError(f"Unknown import thread: {thread_id}") from exc

        prompt = build_import_prompt(
            schedule=schedule,
            week_start=week_start,
            skill_text=self._skill_text or "",
            user_message=user_message,
            is_start=False,
        )
        raw_text = await self._send_to_agent(session=session, prompt=prompt)
        agent = parse_agent_response(raw_text)
        return ImportApiResponse(
            thread_id=thread_id,
            agent=agent,
            raw_text=raw_text,
        )

    async def chat(
        self,
        *,
        message: str,
        schedule: ScheduleState,
        week_start: str | None,
        thread_id: str | None = None,
    ) -> ChatApiResponse:
        if thread_id:
            try:
                session = await self._get_or_resume_session(thread_id)
                is_start = False
            except Exception:
                session = await self._create_agent_session()
                is_start = True
        else:
            session = await self._create_agent_session()
            is_start = True

        prompt = build_chat_prompt(
            schedule=schedule,
            week_start=week_start,
            skill_text=self._skill_text or "",
            user_message=message,
            is_start=is_start,
        )
        raw_text = await self._send_to_agent(session=session, prompt=prompt)
        reply = parse_chat_response(raw_text)
        return ChatApiResponse(
            thread_id=session.thread_id,
            reply=reply,
            raw_text=raw_text,
        )

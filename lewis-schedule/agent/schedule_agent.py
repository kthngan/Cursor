from __future__ import annotations

import json
import re
import uuid
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from cursor_sdk import AsyncClient, LocalAgentOptions, SDKImage, UserMessage

from models import AgentResponse, ImportApiResponse, ScheduleState


COMPOSER_MODEL = "composer-2.5"
SKILL_PATH = ".cursor/skills/lewis-schedule-import/SKILL.md"


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


@dataclass
class ImportSession:
    thread_id: str
    agent: Any
    stack: AsyncExitStack
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


class ScheduleAgentService:
    def __init__(self, *, workspace: str, api_key: str) -> None:
        self.workspace = workspace
        self.api_key = api_key
        self._client: AsyncClient | None = None
        self._bridge_stack: AsyncExitStack | None = None
        self._sessions: dict[str, ImportSession] = {}
        self._skill_text: str | None = None

    @property
    def composer_available(self) -> bool:
        return bool(self.api_key)

    async def startup(self) -> None:
        if not self.api_key:
            return

        self._bridge_stack = AsyncExitStack()
        bridge = await AsyncClient.launch_bridge(workspace=self.workspace)
        self._client = await self._bridge_stack.enter_async_context(bridge)
        self._skill_text = self._read_skill_text()

    async def shutdown(self) -> None:
        for session in list(self._sessions.values()):
            await self.close_session(session.thread_id)
        if self._bridge_stack is not None:
            await self._bridge_stack.aclose()
            self._bridge_stack = None
            self._client = None

    def _read_skill_text(self) -> str:
        from pathlib import Path

        skill_file = Path(self.workspace) / SKILL_PATH
        if skill_file.is_file():
            return skill_file.read_text(encoding="utf-8")
        return "Return schedule import JSON with modes questions, proposal, or noop."

    async def _create_agent_session(self) -> ImportSession:
        if self._client is None:
            raise RuntimeError(
                "Composer is not configured. Set CURSOR_API_KEY in lewis-schedule/agent/.env"
            )

        stack = AsyncExitStack()
        agent_handle = await self._client.agents.create(
            model=COMPOSER_MODEL,
            api_key=self.api_key,
            local=LocalAgentOptions(cwd=self.workspace),
        )
        agent = await stack.enter_async_context(agent_handle)
        thread_id = str(uuid.uuid4())
        session = ImportSession(thread_id=thread_id, agent=agent, stack=stack)
        self._sessions[thread_id] = session
        return session

    async def close_session(self, thread_id: str) -> None:
        session = self._sessions.pop(thread_id, None)
        if session is not None:
            await session.stack.aclose()

    async def _send_to_agent(
        self,
        *,
        session: ImportSession,
        prompt: str,
        image_base64: str | None = None,
        mime_type: str = "image/jpeg",
    ) -> str:
        images = []
        if image_base64:
            images.append(SDKImage.from_data(image_base64, mime_type))

        run = await session.agent.send(
            UserMessage(text=prompt, images=images or None)
        )
        return await run.text()

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
        session = self._sessions.get(thread_id)
        if session is None:
            raise KeyError(f"Unknown import thread: {thread_id}")

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

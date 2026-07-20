import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models import (
    ChatRequest,
    HealthResponse,
    ImportContinueRequest,
    ImportStartRequest,
    ScheduleSaveResponse,
    ScheduleState,
)
from schedule_agent import ScheduleAgentService, monday_of_week
from schedule_store import load_schedule, save_schedule

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
TEMPLATE_PATH = ROOT / "templates" / "default_template.json"
DEFAULT_WORKSPACE = ROOT.parent
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR") or DEFAULT_WORKSPACE).resolve()
API_KEY = os.environ.get("CURSOR_API_KEY", "")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
CLOUD_REPO_URL = os.environ.get("CLOUD_REPO_URL", "https://github.com/kthngan/Cursor")
PORT = int(os.environ.get("PORT", "8790"))
HOST = os.environ.get("HOST", "0.0.0.0")


def verify_token(request: Request) -> None:
    if not ACCESS_TOKEN:
        return
    auth = request.headers.get("authorization", "")
    token = ""
    if auth.lower().startswith("bearer "):
        token = auth[7:].strip()
    if not token:
        token = request.query_params.get("token", "")
    if token != ACCESS_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid access token")


@asynccontextmanager
async def lifespan(app: FastAPI):
    service = ScheduleAgentService(
        workspace=str(WORKSPACE),
        api_key=API_KEY,
        cloud_repo_url=CLOUD_REPO_URL,
    )
    await service.startup()
    app.state.schedule_agent = service
    yield
    await service.shutdown()


app = FastAPI(title="Lewis Schedule", lifespan=lifespan)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health(request: Request) -> HealthResponse:
    verify_token(request)
    service: ScheduleAgentService = request.app.state.schedule_agent
    return HealthResponse(
        ok=True,
        workspace=str(WORKSPACE),
        composer_available=service.composer_available,
        warm_agent_ready=service.warm_agent_ready,
    )


@app.get("/api/template")
async def template(request: Request) -> JSONResponse:
    verify_token(request)
    if not TEMPLATE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Template not found")
    return JSONResponse(json.loads(TEMPLATE_PATH.read_text(encoding="utf-8")))


@app.get("/api/schedule")
async def get_schedule(request: Request, week_start: str) -> JSONResponse:
    verify_token(request)
    try:
        schedule = load_schedule(week_start)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if schedule is None:
        raise HTTPException(status_code=404, detail="No saved schedule for this week")
    return JSONResponse(schedule.model_dump())


@app.put("/api/schedule")
async def put_schedule(request: Request, body: ScheduleState) -> ScheduleSaveResponse:
    verify_token(request)
    if not body.week_start:
        raise HTTPException(status_code=422, detail="week_start is required")
    try:
        week_start = save_schedule(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ScheduleSaveResponse(ok=True, week_start=week_start)


@app.post("/schedule/import/start")
async def import_start(request: Request, body: ImportStartRequest):
    verify_token(request)
    service: ScheduleAgentService = request.app.state.schedule_agent
    if not service.composer_available:
        raise HTTPException(
            status_code=503,
            detail="Cloud agent import is unavailable. Set CURSOR_API_KEY on the server.",
        )
    try:
        result = await service.start_import(
            schedule=body.schedule,
            week_start=body.week_start or monday_of_week(),
            image_base64=body.image_base64,
            mime_type=body.mime_type,
        )
        return result.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/schedule/import/continue")
async def import_continue(request: Request, body: ImportContinueRequest):
    verify_token(request)
    service: ScheduleAgentService = request.app.state.schedule_agent
    if not service.composer_available:
        raise HTTPException(status_code=503, detail="Cloud agent import is unavailable.")
    try:
        result = await service.continue_import(
            thread_id=body.thread_id,
            schedule=body.schedule,
            week_start=body.schedule.week_start or monday_of_week(),
            user_message=body.user_message,
        )
        return result.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.delete("/schedule/import/{thread_id}")
async def import_close(request: Request, thread_id: str):
    verify_token(request)
    service: ScheduleAgentService = request.app.state.schedule_agent
    await service.close_session(thread_id)
    return {"ok": True}


@app.post("/schedule/chat")
async def schedule_chat(request: Request, body: ChatRequest):
    verify_token(request)
    service: ScheduleAgentService = request.app.state.schedule_agent
    if not service.composer_available:
        raise HTTPException(
            status_code=503,
            detail="Chat is unavailable. Set CURSOR_API_KEY on the server.",
        )
    if not body.message.strip():
        raise HTTPException(status_code=422, detail="message is required")
    try:
        result = await service.chat(
            message=body.message.strip(),
            schedule=body.schedule,
            week_start=body.schedule.week_start or monday_of_week(),
            thread_id=body.thread_id,
        )
        return result.model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from models import (
    HealthResponse,
    ImportContinueRequest,
    ImportStartRequest,
)
from schedule_agent import ScheduleAgentService, monday_of_week

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"
TEMPLATE_PATH = ROOT / "templates" / "default_template.json"
DEFAULT_WORKSPACE = ROOT.parent
WORKSPACE = Path(os.environ.get("WORKSPACE_DIR") or DEFAULT_WORKSPACE).resolve()
API_KEY = os.environ.get("CURSOR_API_KEY", "")
ACCESS_TOKEN = os.environ.get("ACCESS_TOKEN", "")
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
    service = ScheduleAgentService(workspace=str(WORKSPACE), api_key=API_KEY)
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
    )


@app.get("/api/template")
async def template(request: Request) -> JSONResponse:
    verify_token(request)
    if not TEMPLATE_PATH.is_file():
        raise HTTPException(status_code=404, detail="Template not found")
    return JSONResponse(json.loads(TEMPLATE_PATH.read_text(encoding="utf-8")))


@app.post("/schedule/import/start")
async def import_start(request: Request, body: ImportStartRequest):
    verify_token(request)
    service: ScheduleAgentService = request.app.state.schedule_agent
    if not service.composer_available:
        raise HTTPException(
            status_code=503,
            detail="Composer import is unavailable. Set CURSOR_API_KEY on the server.",
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
        raise HTTPException(status_code=503, detail="Composer import is unavailable.")
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


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host=HOST, port=PORT, reload=False)

"""HTTP API and the same-origin mobile playground."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

from .laya_mlx_backend import InputTooLong, LayaMLXBackend
from .snake_demo import FinishedGame, SnakeService, UnknownGame, policy_for_backend

JsonValue = str | dict[str, Any] | list[Any]
DEFAULT_MODEL = "laya-multilingual-mlx"


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["choice", "score", "noul"]
    instructions: JsonValue
    criteria: dict[str, Any] | list[Any] | None = None

    @model_validator(mode="after")
    def validate_criteria(self) -> Question:
        if not self.instructions:
            raise ValueError("instructions must not be empty")
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not 2 <= len(self.criteria) <= 255:
                raise ValueError("choice criteria must contain 2 to 255 named options")
        elif self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 10:
                raise ValueError("score criteria must contain 2 to 10 ordered levels")
        elif self.criteria is not None and (
            not isinstance(self.criteria, dict) or not set(self.criteria) <= {"true", "false"}
        ):
            raise ValueError("noul criteria must contain only true/false descriptions")
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    state: JsonValue
    model: str
    questions: dict[str, Question] = Field(min_length=1, max_length=32)

    @model_validator(mode="after")
    def validate_request(self) -> DecisionRequest:
        if not self.model:
            raise ValueError("model must not be empty")
        if any(not key for key in self.questions):
            raise ValueError("question ids must not be empty")
        return self


class SnakeStartRequest(BaseModel):
    model: str = DEFAULT_MODEL


def create_app(backend: Any = None, snake_policy: Any = None,
               backends: dict[str, Any] | None = None,
               snake_policies: dict[str, Any] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup if the checkpoint cannot load. /healthz must never imply a
        # working model before weights are actually available.
        if backends is not None:
            app.state.backends = backends
        elif backend is not None:
            app.state.backends = {backend.model_name: backend}
        else:
            from .laya_coreml_backend import LayaCoreMLBackend
            app.state.backends = {
                DEFAULT_MODEL: LayaMLXBackend(),
                "laya-multilingual-coreml": LayaCoreMLBackend(),
                "laya-multilingual-coreml-ane": LayaCoreMLBackend(
                    "aac6fef/laya-multilingual-coreml-ane"),
            }
        app.state.backend = app.state.backends[DEFAULT_MODEL]
        app.state.inference_lock = Lock()
        policies = snake_policies if snake_policies is not None else {
            name: policy_for_backend(selected)
            for name, selected in app.state.backends.items() if hasattr(selected, "agent")
        }
        if snake_policy is not None:
            policies[DEFAULT_MODEL] = snake_policy
        app.state.snake = SnakeService(policies[DEFAULT_MODEL], policies=policies) if policies else None
        yield

    app = FastAPI(title="Jev System One Local", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def playground():
        return FileResponse(Path(__file__).with_name("playground.html"), media_type="text/html")

    @app.get("/vendor/typesafe-sdk.mjs", include_in_schema=False)
    def official_sdk():
        return FileResponse(Path(__file__).with_name("vendor") / "typesafe-sdk.mjs",
                            media_type="text/javascript")

    @app.get("/snake", include_in_schema=False)
    def snake_page():
        return FileResponse(Path(__file__).with_name("snake.html"), media_type="text/html")

    @app.post("/snake/api/sessions")
    async def start_snake(request: SnakeStartRequest | None = None):
        def start():
            with app.state.inference_lock:
                return app.state.snake.start(model=request.model if request else DEFAULT_MODEL)
        try:
            return await run_in_threadpool(start)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/snake/api/sessions/{session_id}/step")
    async def step_snake(session_id: str):
        def step():
            with app.state.inference_lock:
                return app.state.snake.step(session_id)
        try:
            return await run_in_threadpool(step)
        except UnknownGame as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except FinishedGame as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/healthz")
    def health():
        loaded = app.state.backend
        return {"status": "ready", "backend": loaded.backend_name, "model": loaded.model_name}

    @app.get("/v1/models")
    def models():
        loaded = app.state.backend
        models = []
        for name, selected in [("jev-latest", loaded), *app.state.backends.items()]:
            models.append({"name": name,
                           "description": f"Local {selected.backend_name} checkpoint (Jev API compatible, not Jev weights)",
                           "release_date": "2026-09-19" if selected.backend_name == "laya-mlx" else "2026-09-20",
                           "backend": selected.backend_name,
                           "checkpoint": getattr(selected, "checkpoint", None)})
        return {"models": models}

    @app.post("/v1/systemone")
    async def system_one(request: DecisionRequest):
        name = DEFAULT_MODEL if request.model == "jev-latest" else request.model
        selected = app.state.backends.get(name)
        if selected is None:
            raise HTTPException(status_code=422, detail=f"unsupported model: {request.model}")
        questions = {key: value.model_dump(exclude_none=True) for key, value in request.questions.items()}

        def evaluate():
            with app.state.inference_lock:
                return selected.evaluate(request.state, questions)

        try:
            return await run_in_threadpool(evaluate)
        except (InputTooLong, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app

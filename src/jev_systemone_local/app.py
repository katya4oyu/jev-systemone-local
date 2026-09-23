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
        if self.model not in {"jev-latest", "laya-multilingual-mlx"}:
            raise ValueError("unsupported model; available: jev-latest, laya-multilingual-mlx")
        if any(not key for key in self.questions):
            raise ValueError("question ids must not be empty")
        return self


def create_app(backend: Any = None, snake_policy: Any = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Fail startup if the checkpoint cannot load. /healthz must never imply a
        # working model before weights are actually available.
        app.state.backend = backend if backend is not None else LayaMLXBackend()
        app.state.inference_lock = Lock()
        policy = snake_policy if snake_policy is not None else (
            policy_for_backend(app.state.backend) if hasattr(app.state.backend, "agent") else None
        )
        app.state.snake = SnakeService(policy) if policy is not None else None
        yield

    app = FastAPI(title="Jev System One Local", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def playground():
        return FileResponse(Path(__file__).with_name("playground.html"), media_type="text/html")

    @app.get("/snake", include_in_schema=False)
    def snake_page():
        return FileResponse(Path(__file__).with_name("snake.html"), media_type="text/html")

    @app.post("/snake/api/sessions")
    async def start_snake():
        def start():
            with app.state.inference_lock:
                return app.state.snake.start()
        return await run_in_threadpool(start)

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
        description = "Local Laya-MLX multilingual checkpoint (Jev API compatible, not Jev weights)"
        return {"models": [
            {"name": name, "description": description, "release_date": "2026-09-19",
             "backend": loaded.backend_name, "checkpoint": getattr(loaded, "checkpoint", None)}
            for name in ("jev-latest", loaded.model_name)
        ]}

    @app.post("/v1/systemone")
    async def system_one(request: DecisionRequest):
        questions = {key: value.model_dump(exclude_none=True) for key, value in request.questions.items()}

        def evaluate():
            with app.state.inference_lock:
                return app.state.backend.evaluate(request.state, questions)

        try:
            return await run_in_threadpool(evaluate)
        except (InputTooLong, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app

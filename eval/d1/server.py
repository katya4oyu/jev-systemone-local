"""Task-local HTTP adapter for the official LiquidAI/d1-3B Python decision API.

This is evaluation glue, not a vendor-provided server or a product integration.
"""
from __future__ import annotations

import argparse
import importlib
import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from starlette.concurrency import run_in_threadpool

MODEL_ID = "liquidai-d1-3b-pytorch-mps"
OFFICIAL_MODEL = "LiquidAI/d1-3B"
REVISION = "051bcc464b01b9f92942b364d9586b0ef5912432"
DEFAULT_MODEL_PATH = (
    "/Users/yuya/.cache/huggingface/hub/models--LiquidAI--d1-3B/"
    f"snapshots/{REVISION}"
)

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
        if not self.model:
            raise ValueError("model must not be empty")
        if any(not key for key in self.questions):
            raise ValueError("question ids must not be empty")
        return self


class InputTooLong(ValueError):
    pass


class D1Runtime:
    """Loads and calls only the official checkpoint's specialized System One API."""

    def __init__(self, model: Any, engine: Any, as_question: Any, max_input_tokens: int,
                 device: str = "mps", dtype: str = "torch.bfloat16") -> None:
        self.model = model
        self.engine = engine
        self.as_question = as_question
        self.max_input_tokens = max_input_tokens
        self.device = device
        self.dtype = dtype

    @classmethod
    def load(cls, model_path: str | None = None) -> D1Runtime:
        import torch
        from transformers import AutoModel

        if not torch.backends.mps.is_available():
            raise RuntimeError("PyTorch MPS is unavailable; CPU fallback is forbidden")
        snapshot = os.path.realpath(model_path or os.environ.get("D1_MODEL_PATH", DEFAULT_MODEL_PATH))
        if os.path.basename(snapshot) != REVISION:
            raise RuntimeError(f"expected immutable snapshot {REVISION}, got {snapshot}")
        model = AutoModel.from_pretrained(
            snapshot,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("mps").eval()
        devices = {parameter.device.type for parameter in model.parameters()}
        dtypes = {parameter.dtype for parameter in model.parameters()}
        if devices != {"mps"}:
            raise RuntimeError(f"model parameters are not exclusively on MPS: {devices}")
        if dtypes != {torch.bfloat16}:
            raise RuntimeError(f"model parameters are not exclusively bfloat16: {dtypes}")

        engine = model.engine
        prompt_module = importlib.import_module(
            f"{type(engine).__module__.rsplit('.', 1)[0]}.prompt"
        )
        text_config = model.config.text_config
        max_input_tokens = int(text_config.max_position_embeddings)
        if max_input_tokens != 32768:
            raise RuntimeError(f"unexpected checkpoint context limit: {max_input_tokens}")
        return cls(model, engine, prompt_module.as_question, max_input_tokens)

    def prepared_input_tokens(self, state: JsonValue, questions: dict[str, dict[str, Any]]) -> int:
        typed = [self.as_question(question) for question in questions.values()]
        return int(self.engine.tokens(state, typed))

    def system_one(self, state: JsonValue, questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return self.model.system_one(state, questions)

    def public_identity(self) -> dict[str, Any]:
        return {
            "id": MODEL_ID,
            "official_model": OFFICIAL_MODEL,
            "revision": REVISION,
            "device": self.device,
            "dtype": self.dtype,
            "max_input_tokens": self.max_input_tokens,
        }


def create_app(runtime: Any | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = runtime if runtime is not None else D1Runtime.load()
        app.state.inference_lock = Lock()
        app.state.metrics_lock = Lock()
        app.state.metrics = {
            "systemone_requests": 0,
            "successful_requests": 0,
            "rejected_requests": 0,
            "runtime_errors": 0,
            "questions": 0,
            "question_types": {"choice": 0, "noul": 0, "score": 0},
            "max_prepared_input_tokens": 0,
            "max_response_usage_input_tokens": 0,
        }
        yield

    app = FastAPI(title="LiquidAI d1-3B local evaluation adapter", lifespan=lifespan)

    @app.get("/health")
    @app.get("/healthz")
    def health() -> dict[str, Any]:
        loaded = app.state.runtime
        return {"status": "ready", "model": loaded.public_identity()}

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        return {"models": [{
            "name": MODEL_ID,
            "id": MODEL_ID,
            "description": "Task-local adapter to official LiquidAI/d1-3B; not a vendor server",
            "backend": "pytorch-mps",
            "checkpoint": f"{OFFICIAL_MODEL}@{REVISION}",
        }]}

    @app.get("/_task/stats")
    def stats() -> dict[str, Any]:
        with app.state.metrics_lock:
            return dict(app.state.metrics) | {
                "question_types": dict(app.state.metrics["question_types"]),
                "model": app.state.runtime.public_identity(),
            }

    @app.post("/v1/systemone")
    async def system_one(request: DecisionRequest) -> dict[str, Any]:
        loaded = app.state.runtime
        with app.state.metrics_lock:
            app.state.metrics["systemone_requests"] += 1
        if request.model != MODEL_ID:
            with app.state.metrics_lock:
                app.state.metrics["rejected_requests"] += 1
            raise HTTPException(status_code=422, detail=f"unsupported model: {request.model}")
        questions = {
            name: question.model_dump(exclude_none=True)
            for name, question in request.questions.items()
        }

        def evaluate() -> tuple[int, dict[str, Any]]:
            with app.state.inference_lock:
                prepared = loaded.prepared_input_tokens(request.state, questions)
                if prepared > loaded.max_input_tokens:
                    raise InputTooLong(
                        f"prompt of {prepared} tokens exceeds the context window of "
                        f"{loaded.max_input_tokens} tokens"
                    )
                return prepared, loaded.system_one(request.state, questions)

        try:
            prepared, response = await run_in_threadpool(evaluate)
        except InputTooLong as exc:
            with app.state.metrics_lock:
                app.state.metrics["rejected_requests"] += 1
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except ValueError as exc:
            with app.state.metrics_lock:
                app.state.metrics["rejected_requests"] += 1
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception:
            with app.state.metrics_lock:
                app.state.metrics["runtime_errors"] += 1
            raise

        usage = response.get("usage", {})
        input_used = int(usage.get("input_tokens", 0))
        output_used = int(usage.get("output_tokens", 0))
        with app.state.metrics_lock:
            app.state.metrics["successful_requests"] += 1
            app.state.metrics["questions"] += len(questions)
            for question in request.questions.values():
                app.state.metrics["question_types"][question.type] += 1
            app.state.metrics["max_prepared_input_tokens"] = max(
                app.state.metrics["max_prepared_input_tokens"], prepared
            )
            app.state.metrics["max_response_usage_input_tokens"] = max(
                app.state.metrics["max_response_usage_input_tokens"], input_used
            )
        return dict(response) | {"model": MODEL_ID}

    return app


app = create_app()


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8025)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

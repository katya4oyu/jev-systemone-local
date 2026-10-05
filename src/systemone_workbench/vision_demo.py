"""Separate same-origin FastAPI app for local Clef-Flash image inference."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from .clef_flash_backend import ClefFlashBackend, ContextLimitError, InvalidImage, MAX_IMAGE_BYTES

MAX_REQUEST_BYTES = ((MAX_IMAGE_BYTES + 2) // 3) * 4 + 1024
PAGE = Path(__file__).with_name("vision_demo.html")


def _same_origin(request: Request) -> bool:
    """Reject browser cross-site writes while allowing local non-browser API probes."""
    if request.headers.get("sec-fetch-site", "").lower() == "cross-site":
        return False
    origin = request.headers.get("origin")
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
    except ValueError:
        return False
    return (
        parsed.scheme == request.url.scheme
        and parsed.netloc.casefold() == request.url.netloc.casefold()
        and parsed.path in ("", "/")
        and not parsed.query
        and not parsed.fragment
    )


async def _read_json_limited(request: Request) -> dict[str, Any]:
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise HTTPException(status_code=415, detail="application/json の画像データを送信してください。")
    declared = request.headers.get("content-length")
    if declared:
        try:
            if int(declared) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="画像データが上限を超えています。")
        except ValueError:
            raise HTTPException(status_code=400, detail="Content-Length が不正です。") from None
    chunks: list[bytes] = []
    total = 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_REQUEST_BYTES:
            raise HTTPException(status_code=413, detail="画像データが上限を超えています。")
        chunks.append(chunk)
    try:
        body = json.loads(b"".join(chunks))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HTTPException(status_code=400, detail="JSON を読み取れません。") from None
    if not isinstance(body, dict) or set(body) != {"image", "mime_type"}:
        raise HTTPException(status_code=422, detail="image と mime_type のみを指定してください。")
    if not isinstance(body["image"], str) or not isinstance(body["mime_type"], str):
        raise HTTPException(status_code=422, detail="image と mime_type は文字列で指定してください。")
    return body


def create_vision_app(backend: Any | None = None, checkpoint: str | Path | None = None) -> FastAPI:
    """Build the image-only demo without importing or starting the legacy model app."""
    if backend is not None and checkpoint is not None:
        raise ValueError("pass either a test backend or a local checkpoint, not both")
    owned_backend = backend is None
    selected = backend if backend is not None else ClefFlashBackend(
        checkpoint or Path.home() / ".cache/huggingface/hub/models--mlx-community--clef-flash-4bit/snapshots/140bf7e037f5fa95a96535feca112f46c15927cf"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.vision_backend = selected
        app.state.inference_lock = Lock()
        app.state.vision_ready = not owned_backend
        try:
            if owned_backend:
                await asyncio.get_running_loop().run_in_executor(None, selected.load)
                app.state.vision_ready = True
            yield
        finally:
            if owned_backend:
                selected.close()

    app = FastAPI(title="Clef-Flash local vision demo", lifespan=lifespan)

    @app.get("/", include_in_schema=False)
    def vision_page():
        return FileResponse(PAGE, media_type="text/html")

    @app.get("/vision-camera.mjs", include_in_schema=False)
    def camera_module():
        return FileResponse(PAGE.with_name("vision_camera.mjs"), media_type="text/javascript")

    @app.get("/healthz")
    def health():
        if not app.state.vision_ready or not getattr(app.state.vision_backend, "ready", True):
            raise HTTPException(status_code=503, detail="model is not ready")
        model = app.state.vision_backend
        return {
            "status": "ready",
            "model": {"id": model.model_id, "quantization": model.quantization},
        }

    @app.post("/vision/api/analyze")
    async def analyze(request: Request):
        if not _same_origin(request):
            raise HTTPException(status_code=403, detail="same-origin requests only")
        if not app.state.vision_ready:
            raise HTTPException(status_code=503, detail="model is not ready")
        raw_lock = app.state.inference_lock
        if not raw_lock.acquire(blocking=False):
            raise HTTPException(status_code=503, detail="推論中です。少し待ってから再度お試しください。")
        try:
            body = await _read_json_limited(request)
            backend_for_request = app.state.vision_backend
            if hasattr(backend_for_request, "executor"):
                result = await asyncio.get_running_loop().run_in_executor(
                    backend_for_request.executor,
                    backend_for_request.analyze,
                    body["image"],
                    body["mime_type"],
                )
            else:
                result = await asyncio.get_running_loop().run_in_executor(
                    None,
                    backend_for_request.analyze,
                    body["image"],
                    body["mime_type"],
                )
            return result
        except HTTPException:
            raise
        except InvalidImage as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from None
        except ContextLimitError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from None
        except Exception:
            # Never include the request body, local paths, image name, or model internals.
            raise HTTPException(status_code=500, detail="ローカル推論に失敗しました。別の画像で再度お試しください。") from None
        finally:
            raw_lock.release()

    return app

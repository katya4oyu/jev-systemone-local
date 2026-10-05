from __future__ import annotations

import base64
import io
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from eval.vision.jaffe import make_questions as make_jaffe_questions
from systemone_workbench.clef_flash_backend import (
    ClefFlashBackend,
    InvalidImage,
    decode_image,
    make_questions,
)
from systemone_workbench.vision_demo import create_vision_app


def _question_probabilities(face_visible: bool) -> dict[str, dict[str, float]]:
    probabilities: dict[str, dict[str, float]] = {
        "expression": {
            "neutral": 0.1,
            "happiness": 0.4,
            "sadness": 0.1,
            "surprise": 0.1,
            "anger": 0.1,
            "disgust": 0.1,
            "fear": 0.1,
        },
        "face_visible": {"true": 0.8 if face_visible else 0.2, "false": 0.2 if face_visible else 0.8},
    }
    for code in ("hap", "sad", "sur", "ang", "dis", "fea"):
        probabilities[f"rating_{code}"] = {"0": 0.05, "1": 0.1, "2": 0.25, "3": 0.35, "4": 0.25}
    return probabilities


def _synthetic_png() -> bytes:
    image = Image.new("RGB", (64, 32), (228, 219, 198))
    image.putpixel((12, 12), (20, 70, 45))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image.close()
    return buffer.getvalue()


def test_demo_questions_match_all_seven_jaffe_questions_and_add_only_face_check():
    questions = make_questions()
    assert {key: value for key, value in questions.items() if key != "face_visible"} == make_jaffe_questions()
    assert len(questions) == 8
    assert sum(question["type"] == "choice" for question in questions.values()) == 1
    assert sum(question["type"] == "score" for question in questions.values()) == 6
    assert questions["face_visible"]["type"] == "noul"


def test_decoder_accepts_synthetic_png_and_drops_metadata():
    decoded = decode_image(base64.b64encode(_synthetic_png()).decode("ascii"), "image/png")
    assert decoded.mode == "RGB"
    assert decoded.size == (64, 32)
    assert not decoded.info
    decoded.close()


def test_decoder_rejects_mime_mismatch_and_oversize_before_model_call():
    image_b64 = base64.b64encode(_synthetic_png()).decode("ascii")
    with pytest.raises(InvalidImage, match="JPEG、PNG、WebP"):
        decode_image(image_b64, "image/jpeg")
    oversized = base64.b64encode(b"x" * (4 * 1024 * 1024 + 1)).decode("ascii")
    with pytest.raises(InvalidImage, match="4 MiB"):
        decode_image(oversized, "image/png")


@pytest.mark.parametrize("face_visible", [False, True])
def test_result_gates_expression_output_on_demo_only_face_question(tmp_path: Path, face_visible: bool):
    backend = ClefFlashBackend(tmp_path)
    backend.model_id = "mlx-community/clef-flash-4bit"
    backend.quantization = "4-bit"
    result = backend._format_result(_question_probabilities(face_visible), 123.4)
    assert result["model"] == {"id": "mlx-community/clef-flash-4bit", "quantization": "4-bit"}
    assert result["inference_ms"] == 123.4
    assert result["questions_evaluated"] == 8
    assert result["image_media_confirmed"] is True
    assert result["face_check"]["visible"] is face_visible
    if face_visible:
        assert result["answers"]["expression"]["label"] == "happiness"
        assert len(result["answers"]["expression"]["probabilities"]) == 7
        assert len(result["answers"]["scores"]) == 6
        assert result["answers"]["scores"]["HAP"]["value"] == 3.65
    else:
        assert "answers" not in result


class FakeBackend:
    model_id = "local-fake-for-tests"
    quantization = "test-only"
    ready = True

    def __init__(self) -> None:
        self.received: list[tuple[bytes, str]] = []

    def analyze(self, image_b64: str, mime_type: str) -> dict[str, Any]:
        self.received.append((base64.b64decode(image_b64, validate=True), mime_type))
        return {
            "model": {"id": self.model_id, "quantization": self.quantization},
            "inference_ms": 1.0,
            "face_check": {"visible": False, "probabilities": {"true": 0.1, "false": 0.9}},
        }


def test_http_demo_serves_page_health_and_forwards_one_image_without_filename():
    backend = FakeBackend()
    payload = _synthetic_png()
    encoded = base64.b64encode(payload).decode("ascii")
    with TestClient(create_vision_app(backend=backend)) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "CLEF-FLASH" in page.text
        module = client.get("/vision-camera.mjs")
        assert module.status_code == 200
        assert module.headers["content-type"].startswith("text/javascript")
        health = client.get("/healthz")
        assert health.json() == {
            "status": "ready",
            "model": {"id": "local-fake-for-tests", "quantization": "test-only"},
        }
        result = client.post(
            "/vision/api/analyze",
            headers={"Origin": "http://testserver"},
            json={"image": encoded, "mime_type": "image/png"},
        )
        assert result.status_code == 200
        assert result.json()["face_check"]["visible"] is False
        assert backend.received == [(payload, "image/png")]
        forbidden = client.post(
            "/vision/api/analyze",
            headers={"Origin": "http://testserver"},
            json={"image": encoded, "mime_type": "image/png", "filename": "private-face.png"},
        )
        assert forbidden.status_code == 422
        assert client.post(
            "/vision/api/analyze",
            headers={"Origin": "https://attacker.invalid"},
            json={"image": encoded, "mime_type": "image/png"},
        ).status_code == 403
        assert len(backend.received) == 1


def test_http_demo_enforces_request_size_and_recovers_after_bad_request():
    backend = FakeBackend()
    with TestClient(create_vision_app(backend=backend)) as client:
        too_large = "A" * (((4 * 1024 * 1024 + 2) // 3) * 4 + 2048)
        response = client.post(
            "/vision/api/analyze",
            headers={"Origin": "http://testserver"},
            json={"image": too_large, "mime_type": "image/png"},
        )
        assert response.status_code == 413
        good = client.post(
            "/vision/api/analyze",
            headers={"Origin": "http://testserver"},
            json={"image": base64.b64encode(_synthetic_png()).decode("ascii"), "mime_type": "image/png"},
        )
        assert good.status_code == 200
        assert len(backend.received) == 1

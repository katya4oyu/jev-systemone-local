"""Local Clef-Flash vision inference for the isolated image demo."""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000
MAX_IMAGE_SIDE = 2048
SCORE_LEVELS = (
    "表情としてほとんど見えない",
    "弱く見える",
    "ある程度見える",
    "はっきり見える",
    "非常にはっきり見える",
)
EMOTIONS = {
    "neutral": "無表情に近い表情",
    "happiness": "喜びの表情",
    "sadness": "悲しみの表情",
    "surprise": "驚きの表情",
    "anger": "怒りの表情",
    "disgust": "嫌悪の表情",
    "fear": "恐れの表情",
}
RATINGS = {
    "HAP": "喜び",
    "SAD": "悲しみ",
    "SUR": "驚き",
    "ANG": "怒り",
    "DIS": "嫌悪",
    "FEA": "恐れ",
}
ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class InvalidImage(ValueError):
    """The uploaded bytes are not a supported, bounded, single-frame image."""


class ContextLimitError(ValueError):
    """The model rejected an over-length vision request."""


def make_questions() -> dict[str, dict[str, Any]]:
    """The seven JAFFE questions plus one demo-only visual face-presence check."""
    questions: dict[str, dict[str, Any]] = {
        "expression": {
            "type": "choice",
            "instructions": "顔画像に見える表情として最も近いものはどれですか。本人の実際の気持ちは推測せず、見た目だけで選んでください。",
            "criteria": EMOTIONS,
        }
    }
    for code, name in RATINGS.items():
        questions[f"rating_{code.lower()}"] = {
            "type": "score",
            "instructions": f"顔画像に見える{name}の表情成分は、どの程度ですか。本人の気持ちではなく、見た目だけを評価してください。",
            "criteria": list(SCORE_LEVELS),
        }
    questions["face_visible"] = {
        "type": "noul",
        "instructions": "画像に、評価対象として一人の顔がはっきり写っていますか。顔の人数や画質を保証する顔検出ではなく、見た目から判別困難なら false を選んでください。",
        "criteria": {
            "true": "一人の顔がはっきり見える",
            "false": "顔がない、複数ある、または一人の顔がはっきり見えるか判別困難",
        },
    }
    return questions


def decode_image(encoded: str, claimed_mime: str | None) -> Any:
    """Decode a single bounded JPEG/PNG/WebP to metadata-free, oriented RGB pixels."""
    if not isinstance(encoded, str) or not encoded or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise InvalidImage("画像は4 MiB以下で指定してください。")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise InvalidImage("画像データを読み取れません。別の画像を選んでください。") from None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise InvalidImage("画像は4 MiB以下で指定してください。")

    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(raw)) as source:
            mime = ALLOWED_FORMATS.get(source.format or "")
            if mime is None or (claimed_mime and claimed_mime != mime):
                raise InvalidImage("JPEG、PNG、WebPの画像を選んでください。")
            if getattr(source, "n_frames", 1) != 1:
                raise InvalidImage("アニメーション画像には対応していません。静止画像を選んでください。")
            width, height = source.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise InvalidImage("画像は1,600万画素以下にしてください。")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise InvalidImage("画像データを読み取れません。別の画像を選んでください。") from None

    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    image.info.clear()
    return image


def _checkpoint_identity(path: Path) -> tuple[str, str]:
    """Return a non-path model ID and quantization label for a local HF snapshot."""
    parent = path.parent.parent
    if path.parent.name == "snapshots" and parent.name.startswith("models--"):
        model_id = parent.name.removeprefix("models--").replace("--", "/")
    else:
        model_id = path.name
    match = re.search(r"(?<!\d)([48])[-_]?bit(?![a-z])", model_id, re.IGNORECASE)
    quantization = f"{match.group(1)}-bit" if match else "不明"
    return model_id, quantization


class ClefFlashBackend:
    """Load one explicitly selected local checkpoint on a dedicated serial MLX thread."""

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = Path(checkpoint).expanduser()
        self.model_id, self.quantization = _checkpoint_identity(self.checkpoint)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clef-mlx")
        self._model: Any = None
        self._mx: Any = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready and self._model is not None

    @property
    def executor(self) -> ThreadPoolExecutor:
        return self._executor

    def load(self) -> None:
        """Perform import and load inside the same worker that will run inference."""
        self._executor.submit(self._load_model).result()

    def _load_model(self) -> None:
        checkpoint = self.checkpoint.resolve(strict=True)
        if not checkpoint.is_dir():
            raise FileNotFoundError("--vision-checkpoint must name an existing local directory")
        sys.path.insert(0, str(checkpoint))
        import clef_mlx  # trusted code from the explicitly selected local checkpoint
        import mlx.core as mx

        model = clef_mlx.load(str(checkpoint), backend="vlm")
        if not getattr(model, "vision", False) or getattr(model, "processor", None) is None:
            raise RuntimeError("selected checkpoint does not have a vision-enabled processor")
        self._model = model
        self._mx = mx
        self._ready = True

    def analyze(self, image_b64: str, mime_type: str | None) -> dict[str, Any]:
        """Run one fixed-schema request; do not retain or log the image or filename."""
        if not self.ready:
            raise RuntimeError("Clef-Flash model is not ready")
        image = decode_image(image_b64, mime_type)
        try:
            record = {
                "state": "添付された顔画像に表れている表情を、見た目だけで評価してください。人物の特定、性格、実際の感情は推測しないでください。",
                "questions": make_questions(),
                "images": [image],
            }
            started = time.perf_counter()
            try:
                encoded, logits = self._model.logits(record, truncate=False)
            except Exception as exc:
                if type(exc).__name__ == "ContextTooLong":
                    raise ContextLimitError("画像を含む入力がモデルの長さ上限を超えました。") from None
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000
            media = encoded.media or {}
            if "pixel_values" not in media:
                raise RuntimeError("image data did not reach the model vision processor")
            probabilities: dict[str, dict[str, float]] = {}
            for question, values in zip(encoded.questions, logits):
                probs = self._mx.softmax(values.astype(self._mx.float32)).tolist()
                probabilities[question.question_id] = {
                    option_id: float(probability)
                    for option_id, probability in zip(question.option_ids, probs)
                }
            return self._format_result(probabilities, elapsed_ms)
        finally:
            image.close()

    def _format_result(self, probabilities: dict[str, dict[str, float]], elapsed_ms: float) -> dict[str, Any]:
        expected_ids = set(make_questions())
        if set(probabilities) != expected_ids:
            raise RuntimeError("model response did not include the complete fixed question set")
        face = probabilities["face_visible"]
        face_probs = {key: face[key] for key in ("true", "false")}
        if any(not math.isfinite(value) or value < 0 for value in face_probs.values()):
            raise RuntimeError("model returned invalid face-check probabilities")
        visible = face_probs["true"] >= face_probs["false"]
        result: dict[str, Any] = {
            "model": {"id": self.model_id, "quantization": self.quantization},
            "inference_ms": round(elapsed_ms, 1),
            "questions_evaluated": len(probabilities),
            "image_media_confirmed": True,
            "face_check": {"visible": visible, "probabilities": face_probs},
        }
        if not visible:
            return result

        choice = probabilities["expression"]
        if set(choice) != set(EMOTIONS):
            raise RuntimeError("model returned an incomplete expression distribution")
        if any(not math.isfinite(value) or value < 0 for value in choice.values()):
            raise RuntimeError("model returned invalid expression probabilities")
        result["answers"] = {
            "expression": {"label": max(choice, key=choice.__getitem__), "probabilities": choice}
        }
        scores: dict[str, dict[str, Any]] = {}
        for code, name in RATINGS.items():
            distribution = probabilities[f"rating_{code.lower()}"]
            keys = [str(i) for i in range(5)]
            if any(key not in distribution for key in keys):
                raise RuntimeError("model returned an incomplete score distribution")
            values = [distribution[key] for key in keys]
            if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
                raise RuntimeError("model returned invalid score probabilities")
            score = 1 + sum(index * value for index, value in enumerate(values)) / sum(values)
            scores[code] = {"name": name, "value": round(score, 2)}
        result["answers"]["scores"] = scores
        return result

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

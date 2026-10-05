"""Local-only JAFFE image evaluation for fixed Clef-Flash MLX checkpoints.

The dataset is read from a user-provided local directory. Images and per-image
answers are never written to the result file; only aggregate metrics are saved.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from PIL import Image

POSE_TO_LABEL = {
    "NE": "neutral",
    "HA": "happiness",
    "SA": "sadness",
    "SU": "surprise",
    "AN": "anger",
    "DI": "disgust",
    "FE": "fear",
}
LABELS = tuple(POSE_TO_LABEL.values())
RATING_DIMENSIONS = {
    "HAP": "happiness",
    "SAD": "sadness",
    "SUR": "surprise",
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
}
RATING_ORDER = tuple(RATING_DIMENSIONS)
PRIMARY_TABLE_END = "Semantic Ratings Data (Fear Excluded)"
EXPECTED_ARCHIVE_BYTES = 12_290_558
EXPECTED_ARCHIVE_MD5 = "fe13f3302eb9968ef04367456f665436"
EXPECTED_README_BYTES = 18_459
EXPECTED_README_MD5 = "3d3e20343692832266cb094f0c54c5c0"
IMAGE_NAME = re.compile(r"([A-Z]{2})\.([A-Z]{2}[1-5])\.\d+")
RATING_ID = re.compile(r"[A-Z]{2}-[A-Z]{2}[1-5]")

SNAPSHOTS = {
    4: {
        "id": "mlx-community/clef-flash-4bit",
        "revision": "140bf7e037f5fa95a96535feca112f46c15927cf",
        "path": Path.home() / ".cache/huggingface/hub/models--mlx-community--clef-flash-4bit/snapshots/140bf7e037f5fa95a96535feca112f46c15927cf",
    },
    8: {
        "id": "mlx-community/clef-flash-8bit",
        "revision": "dfa0993decb4f8507a0eae01afd1b2d33a4bb734",
        "path": Path.home() / ".cache/huggingface/hub/models--mlx-community--clef-flash-8bit/snapshots/dfa0993decb4f8507a0eae01afd1b2d33a4bb734",
    },
}
DEFAULT_DATA = Path.home() / ".hermes/cache/scratch/jaffe-official-14974867/extracted"
DEFAULT_README = DEFAULT_DATA.parent / "README_FIRST.txt"
DEFAULT_OUTPUT = Path(__file__).with_name("results")
DEFAULT_LOG = Path.home() / ".hermes/cache/scratch"
SCORE_LEVELS = [
    "表情としてほとんど見えない",
    "弱く見える",
    "ある程度見える",
    "はっきり見える",
    "非常にはっきり見える",
]


@dataclass(frozen=True)
class Sample:
    path: Path
    source_id: str
    pose_code: str
    gold_label: str
    human_ratings: tuple[float, ...] | None


def parse_primary_ratings(readme_text: str) -> dict[str, tuple[float, ...]]:
    """Parse only the six-dimension, full primary study table; preserve source IDs."""
    if PRIMARY_TABLE_END not in readme_text:
        raise ValueError("README is missing the fear-excluded table boundary")
    primary = readme_text.split(PRIMARY_TABLE_END, 1)[0]
    rows: dict[str, tuple[float, ...]] = {}
    for line_number, line in enumerate(primary.splitlines(), start=1):
        parts = line.split()
        if not parts or not parts[0].isdigit():
            continue
        if len(parts) != 8 or not RATING_ID.fullmatch(parts[-1]):
            raise ValueError(f"malformed numeric row in primary ratings table at line {line_number}")
        try:
            values = tuple(float(value) for value in parts[1:7])
        except ValueError as exc:
            raise ValueError(f"non-numeric rating at line {line_number}") from exc
        if len(values) != 6 or any(not math.isfinite(value) or not 1 <= value <= 5 for value in values):
            raise ValueError(f"ratings outside the documented 1..5 scale at line {line_number}")
        source_id = parts[-1]
        if source_id in rows:
            raise ValueError(f"duplicate source rating ID: {source_id}")
        rows[source_id] = values
    if not rows:
        raise ValueError("no rows found in primary ratings table")
    return rows


def normalize_image_key(filename: str | Path) -> str | None:
    """Read the documented person/pose token from an official dotted TIFF name."""
    stem = Path(filename).stem
    match = IMAGE_NAME.fullmatch(stem)
    if not match:
        return None
    return f"{match.group(1)}-{match.group(2)}"


def build_inventory(image_paths: Sequence[Path], ratings: dict[str, tuple[float, ...]]) -> dict[str, Any]:
    """Join image IDs to primary ratings without repairing mismatches."""
    samples: list[Sample] = []
    seen: set[str] = set()
    for path in sorted(image_paths):
        source_id = normalize_image_key(path.name)
        if source_id is None:
            raise ValueError(f"unrecognized JAFFE image filename: {path.name}")
        if source_id in seen:
            raise ValueError(f"duplicate image ID: {source_id}")
        seen.add(source_id)
        pose_code = source_id.split("-", 1)[1][:2]
        if pose_code not in POSE_TO_LABEL:
            raise ValueError(f"unrecognized pose code in source ID: {source_id}")
        samples.append(Sample(path, source_id, pose_code, POSE_TO_LABEL[pose_code], ratings.get(source_id)))
    if not samples:
        raise ValueError("no TIFF images found")
    rating_ids = set(ratings)
    image_ids = seen
    return {
        "samples": samples,
        "rating_row_count": len(ratings),
        "matched_rating_count": len(image_ids & rating_ids),
        "image_without_rating_ids": sorted(image_ids - rating_ids),
        "rating_without_image_ids": sorted(rating_ids - image_ids),
    }


def classification_metrics(
    golds: Sequence[str], predictions: Sequence[str | None], labels: Sequence[str] = LABELS,
) -> dict[str, Any]:
    """Compute aggregate metrics; invalid/missing predictions remain denominator errors."""
    if len(golds) != len(predictions):
        raise ValueError("gold/prediction lengths differ")
    labels = tuple(labels)
    confusion = {gold: {pred: 0 for pred in labels} for gold in labels}
    per_label: dict[str, Any] = {}
    valid = 0
    correct = 0
    for gold, pred in zip(golds, predictions):
        if gold not in confusion:
            raise ValueError(f"gold label outside the fixed label set: {gold}")
        if pred not in labels:
            continue
        valid += 1
        confusion[gold][pred] += 1
        correct += pred == gold
    for label in labels:
        tp = confusion[label][label]
        support = sum(confusion[label].values()) + sum(
            1 for gold, pred in zip(golds, predictions) if gold == label and pred not in labels
        )
        predicted = sum(confusion[gold][label] for gold in labels)
        fp = predicted - tp
        fn = support - tp
        precision = tp / predicted if predicted else None
        recall = tp / support if support else None
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else None
        per_label[label] = {
            "support": support,
            "predicted": predicted,
            "precision": _round_or_none(precision),
            "recall": _round_or_none(recall),
            "f1": _round_or_none(f1),
        }
    f1_values = [v["f1"] for v in per_label.values() if v["f1"] is not None]
    total = len(golds)
    return {
        "n_total": total,
        "n_valid": valid,
        "errors": total - valid,
        "accuracy": _round_or_none(correct / total if total else None),
        "macro_f1": _round_or_none(sum(f1_values) / len(f1_values) if f1_values else None),
        "confusion": confusion,
        "per_label": per_label,
    }


def rating_metrics(golds: Sequence[float], predictions: Sequence[float | None]) -> dict[str, Any]:
    """Compare human 1..5 means with model's zero-based score plus one."""
    if len(golds) != len(predictions):
        raise ValueError("gold/prediction lengths differ")
    pairs: list[tuple[float, float]] = []
    for gold, prediction in zip(golds, predictions):
        if prediction is None or isinstance(prediction, bool):
            continue
        value = float(prediction)
        if not math.isfinite(value) or not 0 <= value <= 4:
            continue
        pairs.append((float(gold), value + 1.0))
    n_total = len(golds)
    errors = n_total - len(pairs)
    mae = sum(abs(gold - predicted) for gold, predicted in pairs) / len(pairs) if pairs else None
    pearson = _pearson([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 2 else None
    return {
        "n_total": n_total,
        "n_valid": len(pairs),
        "errors": errors,
        "mae": _round_or_none(mae),
        "pearson_r": _round_or_none(pearson),
    }


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)
    dx = [x - x_bar for x in xs]
    dy = [y - y_bar for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return None
    return sum(x * y for x, y in zip(dx, dy)) / denom


def _round_or_none(value: float | None) -> float | None:
    return round(value, 6) if value is not None and math.isfinite(value) else None


def make_questions() -> dict[str, dict[str, Any]]:
    """Fixed Japanese expression-appearance questions; no dataset metadata enters."""
    expressions = {
        "neutral": "無表情に近い表情",
        "happiness": "喜びの表情",
        "sadness": "悲しみの表情",
        "surprise": "驚きの表情",
        "anger": "怒りの表情",
        "disgust": "嫌悪の表情",
        "fear": "恐れの表情",
    }
    questions: dict[str, dict[str, Any]] = {
        "expression": {
            "type": "choice",
            "instructions": "顔画像に見える表情として最も近いものはどれですか。本人の実際の気持ちは推測せず、見た目だけで選んでください。",
            "criteria": expressions,
        }
    }
    dimensions = {
        "HAP": "喜び", "SAD": "悲しみ", "SUR": "驚き",
        "ANG": "怒り", "DIS": "嫌悪", "FEA": "恐れ",
    }
    for code, name in dimensions.items():
        questions[f"rating_{code.lower()}"] = {
            "type": "score",
            "instructions": f"顔画像に見える{name}の表情成分は、どの程度ですか。本人の気持ちではなく、見た目だけを評価してください。",
            "criteria": SCORE_LEVELS,
        }
    return questions


def make_record(image: Image.Image | None) -> dict[str, Any]:
    """The state is invariant and deliberately excludes filenames, poses, and ratings."""
    record: dict[str, Any] = {
        "state": "添付された顔画像に表れている表情を、見た目だけで評価してください。人物の特定、性格、実際の感情は推測しないでください。",
        "questions": make_questions(),
    }
    if image is not None:
        record["images"] = [image]
    return record


def load_grayscale_image(path: Path) -> Image.Image:
    """Validate the source is 8-bit grayscale (including grayscale TIFF palettes)."""
    from PIL import Image

    with Image.open(path) as source:
        bits = source.tag_v2.get(258, 8)
        bits = tuple(bits) if isinstance(bits, (tuple, list)) else (bits,)
        if not bits or any(int(value) != 8 for value in bits):
            raise ValueError("JAFFE source image is not 8-bit")
        if source.mode == "P":
            palette = source.getpalette()
            if not palette or len(palette) % 3 or any(
                palette[index] != palette[index + 1] or palette[index] != palette[index + 2]
                for index in range(0, len(palette), 3)
            ):
                raise ValueError("paletted JAFFE image is not grayscale")
        if source.mode not in {"1", "L", "P", "I;16", "I", "F"}:
            raise ValueError(f"unexpected non-grayscale JAFFE image mode: {source.mode}")
        source.load()
        return source.convert("L").convert("RGB")


def discover_images(data_dir: Path) -> list[Path]:
    paths = sorted(
        path for path in data_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if len(paths) != 213:
        raise ValueError(f"expected 213 official JAFFE TIFFs, found {len(paths)}")
    return paths


def log_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"at": datetime.now().astimezone().isoformat(timespec="seconds"), "event": event, **fields}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _array_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return [int(size) for size in shape] if shape is not None else None


def infer(model: Any, record: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, Any], float]:
    """One fixed-schema forward pass, including image preprocessing and the joint head."""
    import mlx.core as mx

    started = time.perf_counter()
    encoded, logits = model.logits(record, truncate=False)
    elapsed_ms = (time.perf_counter() - started) * 1000
    probabilities: dict[str, dict[str, float]] = {}
    for question, values in zip(encoded.questions, logits):
        probs = mx.softmax(values.astype(mx.float32)).tolist()
        probabilities[question.question_id] = {
            option_id: float(probability)
            for option_id, probability in zip(question.option_ids, probs)
        }
    media = encoded.media or {}
    info = {
        "input_tokens": len(encoded.input_ids),
        "has_vision_media": "pixel_values" in media,
        "pixel_values_shape": _array_shape(media.get("pixel_values")),
        "image_grid_thw_shape": _array_shape(media.get("image_grid_thw")),
    }
    return probabilities, info, elapsed_ms


def _interpret(probabilities: dict[str, dict[str, float]]) -> dict[str, Any]:
    choice = probabilities["expression"]
    label = max(choice, key=choice.__getitem__)
    scores: dict[str, float] = {}
    for code in RATING_ORDER:
        distribution = probabilities[f"rating_{code.lower()}"]
        keys = [str(i) for i in range(5)]
        if any(key not in distribution for key in keys):
            raise ValueError("score answer is missing a scale option")
        values = [float(distribution[key]) for key in keys]
        if any(not math.isfinite(value) or value < 0 for value in values) or sum(values) <= 0:
            raise ValueError("score answer contains invalid probabilities")
        scores[code] = sum(index * value for index, value in enumerate(values)) / sum(values)
    return {"label": label, "scores": scores}


def run_smoke(model: Any, samples: Sequence[Sample]) -> dict[str, Any]:
    by_pose: dict[str, Sample] = {}
    for sample in samples:
        if sample.pose_code in {"NE", "HA"} and sample.pose_code not in by_pose:
            by_pose[sample.pose_code] = sample
    if set(by_pose) != {"NE", "HA"}:
        raise ValueError("smoke test needs at least one neutral-pose and one happiness-pose image")
    outputs: dict[str, dict[str, Any]] = {}
    case_info: dict[str, Any] = {}
    for name, sample in (("posed_neutral", by_pose["NE"]), ("posed_happiness", by_pose["HA"])):
        image = load_grayscale_image(sample.path)
        try:
            probs, info, elapsed = infer(model, make_record(image))
        finally:
            image.close()
        if not info["has_vision_media"] or info["pixel_values_shape"] is None:
            raise RuntimeError("real-image smoke did not reach the vision processor")
        outputs[name] = _interpret(probs)
        case_info[name] = {**info, "elapsed_ms": round(elapsed, 3)}
    no_image_probs, no_image_info, no_image_elapsed = infer(model, make_record(None))
    if no_image_info["has_vision_media"]:
        raise RuntimeError("image-absent control unexpectedly included vision pixels")
    outputs["image_absent_control"] = _interpret(no_image_probs)
    case_info["image_absent_control"] = {**no_image_info, "elapsed_ms": round(no_image_elapsed, 3)}
    posed_diff = sum(
        outputs["posed_neutral"]["scores"][key] != outputs["posed_happiness"]["scores"][key]
        for key in RATING_ORDER
    )
    return {
        "same_fixed_state_and_seven_questions": True,
        "cases": case_info,
        "posed_image_choice_differs": outputs["posed_neutral"]["label"] != outputs["posed_happiness"]["label"],
        "posed_image_rating_dimensions_differ": posed_diff,
        "neutral_vs_image_absent_choice_differs": outputs["posed_neutral"]["label"] != outputs["image_absent_control"]["label"],
        "happiness_vs_image_absent_choice_differs": outputs["posed_happiness"]["label"] != outputs["image_absent_control"]["label"],
    }


def _runtime_versions() -> dict[str, str | None]:
    packages = ("mlx", "mlx-vlm", "mlx-lm", "transformers", "Pillow", "numpy")
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def _latency(model: Any, sample: Sample, log_path: Path) -> dict[str, Any]:
    image = load_grayscale_image(sample.path)
    try:
        record = make_record(image)
        warmups = 5
        for index in range(warmups):
            infer(model, record)
            if index == 0 or (index + 1) % 5 == 0:
                log_event(log_path, "latency_warmup", completed=index + 1, total=warmups)
        samples_ms = []
        for index in range(40):
            _, info, elapsed = infer(model, record)
            if not info["has_vision_media"]:
                raise RuntimeError("latency record did not include image media")
            samples_ms.append(elapsed)
            if (index + 1) % 10 == 0:
                log_event(log_path, "latency_measurement", completed=index + 1, total=40)
    finally:
        image.close()
    ordered = sorted(samples_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "scope": "single preloaded image; MLX image processor + vision/backbone + joint head; wall clock, excludes model loading and TIFF file I/O/PIL grayscale-to-RGB conversion",
        "warmup_count": warmups,
        "measurement_count": len(ordered),
        "p50_ms": round((ordered[19] + ordered[20]) / 2, 3),
        "p95_ms": round(ordered[p95_index], 3),
    }


def evaluate(model: Any, samples: Sequence[Sample], log_path: Path) -> dict[str, Any]:
    gold_labels: list[str] = []
    predicted_labels: list[str | None] = []
    score_predictions: dict[str, list[float | None]] = {key: [] for key in RATING_ORDER}
    elapsed_ms: list[float] = []
    input_tokens: list[int] = []
    pixel_shapes: Counter[str] = Counter()
    image_grid_shapes: Counter[str] = Counter()
    vision_media_requests = 0
    request_errors: Counter[str] = Counter()
    output_errors = 0
    for index, sample in enumerate(samples, start=1):
        gold_labels.append(sample.gold_label)
        try:
            image = load_grayscale_image(sample.path)
            try:
                probabilities, info, elapsed = infer(model, make_record(image))
            finally:
                image.close()
            if not info["has_vision_media"]:
                raise RuntimeError("vision media absent for a TIFF input")
            elapsed_ms.append(elapsed)
            vision_media_requests += 1
            input_tokens.append(int(info["input_tokens"]))
            pixel_shapes["x".join(map(str, info["pixel_values_shape"]))] += 1
            image_grid_shapes["x".join(map(str, info["image_grid_thw_shape"]))] += 1
            interpretation = _interpret(probabilities)
            predicted = interpretation["label"]
            if predicted not in LABELS:
                predicted = None
                output_errors += 1
            predicted_labels.append(predicted)
            for code in RATING_ORDER:
                score_predictions[code].append(interpretation["scores"].get(code))
        except Exception as error:  # count and continue; never persist the source ID or payload
            request_errors[type(error).__name__] += 1
            predicted_labels.append(None)
            for code in RATING_ORDER:
                score_predictions[code].append(None)
        if index % 10 == 0 or index == len(samples):
            log_event(log_path, "image_evaluation", completed=index, total=len(samples), request_errors=sum(request_errors.values()))
            print(f"dataset {index}/{len(samples)}; request_errors={sum(request_errors.values())}", flush=True)

    classification = classification_metrics(gold_labels, predicted_labels)
    pose_metrics: dict[str, Any] = {}
    for pose, label in POSE_TO_LABEL.items():
        indexes = [i for i, sample in enumerate(samples) if sample.gold_label == label]
        pose_metrics[pose] = {
            "n_total": len(indexes),
            "errors": sum(predicted_labels[i] not in LABELS for i in indexes),
            "accuracy": _round_or_none(
                sum(predicted_labels[i] == label for i in indexes) / len(indexes) if indexes else None
            ),
        }
    dimension_metrics: dict[str, Any] = {}
    all_human: list[float] = []
    all_model: list[float | None] = []
    for code in RATING_ORDER:
        indexes = [i for i, sample in enumerate(samples) if sample.human_ratings is not None]
        human = [samples[i].human_ratings[RATING_ORDER.index(code)] for i in indexes]  # type: ignore[index]
        model_raw = [score_predictions[code][i] for i in indexes]
        metrics = rating_metrics(human, model_raw)
        predicted_values = [float(value) + 1 for value in model_raw if value is not None and math.isfinite(value)]
        dimension_metrics[code] = {
            **metrics,
            "human_mean": _round_or_none(sum(human) / len(human) if human else None),
            "model_mean": _round_or_none(sum(predicted_values) / len(predicted_values) if predicted_values else None),
            "rating_name": RATING_DIMENSIONS[code],
        }
        all_human.extend(human)
        all_model.extend(model_raw)
    overall_rating_metrics = rating_metrics(all_human, all_model)
    ordered_elapsed = sorted(elapsed_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered_elapsed)) - 1)
    suite_timing = {
        "n_successful_requests": len(elapsed_ms),
        "n_total_requests": len(samples),
        "request_errors": sum(request_errors.values()),
        "vision_media_requests": vision_media_requests,
        "input_tokens": {
            "total": sum(input_tokens),
            "min": min(input_tokens) if input_tokens else None,
            "max": max(input_tokens) if input_tokens else None,
            "distinct_count": len(set(input_tokens)),
        },
        "pixel_values_shape_counts": dict(sorted(pixel_shapes.items())),
        "image_grid_thw_shape_counts": dict(sorted(image_grid_shapes.items())),
        "scope": "one request/image with seven schema questions; includes model image-processor preprocessing and forward pass, excludes model loading and TIFF file I/O/PIL grayscale-to-RGB conversion",
        "total_seconds": round(sum(elapsed_ms) / 1000, 3),
        "p50_ms": round(ordered_elapsed[len(ordered_elapsed) // 2], 3) if ordered_elapsed else None,
        "p95_ms": round(ordered_elapsed[p95_index], 3) if ordered_elapsed else None,
    }
    pose_counts = Counter(sample.gold_label for sample in samples)
    return {
        "classification": {
            **classification,
            "per_pose": pose_metrics,
            "gold_pose_counts": {label: pose_counts[label] for label in LABELS},
            "majority_class_baseline_accuracy": _round_or_none(max(pose_counts.values()) / len(samples)),
            "prediction_errors_are_in_accuracy_denominator": True,
        },
        "human_ratings": {
            "scale": "model expected score 0..4 plus one; README human mean ratings 1..5",
            "dimensions": dimension_metrics,
            "all_matched_dimensions": overall_rating_metrics,
        },
        "timing": {"full_suite": suite_timing},
        "errors": {
            "request_errors_by_exception_type": dict(sorted(request_errors.items())),
            "invalid_choice_outputs": output_errors,
        },
    }


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args(argv)
    config = SNAPSHOTS[args.bits]
    snapshot: Path = config["path"]
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned local model snapshot not found: {snapshot}")
    if not args.readme.is_file():
        raise FileNotFoundError(f"official README_FIRST.txt not found: {args.readme}")
    archive_path = args.archive or args.readme.parent / "jaffe.zip"
    if not archive_path.is_file() or archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
        raise ValueError("official JAFFE archive is missing or has the wrong byte size")
    archive_md5 = md5sum(archive_path)
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError("official JAFFE archive MD5 does not match Zenodo record metadata")
    readme_bytes = args.readme.read_bytes()
    if len(readme_bytes) != EXPECTED_README_BYTES or hashlib.md5(readme_bytes).hexdigest() != EXPECTED_README_MD5:
        raise ValueError("README_FIRST.txt size/MD5 does not match Zenodo record metadata")
    images = discover_images(args.data_dir)
    ratings = parse_primary_ratings(readme_bytes.decode("utf-8"))
    inventory = build_inventory(images, ratings)
    samples: list[Sample] = inventory["samples"]
    if len(samples) != 213 or inventory["rating_row_count"] != 219 or inventory["matched_rating_count"] != 212:
        raise ValueError("dataset inventory differs from verified 213-image / 219-row / 212-rating join")
    log_path = args.log or DEFAULT_LOG / f"jaffe-clef-{args.bits}bit.jsonl"
    output_path = args.output or DEFAULT_OUTPUT / f"clef-flash-{args.bits}bit.json"
    log_event(log_path, "run_start", bits=args.bits, smoke_only=args.smoke_only, image_count=len(samples))
    sys.path.insert(0, str(snapshot))
    import clef_mlx
    import mlx.core as mx

    model = clef_mlx.load(str(snapshot), backend="vlm")
    if not model.vision or model.processor is None:
        raise RuntimeError("pinned checkpoint did not load the vision-enabled processor")
    smoke = run_smoke(model, samples)
    log_event(log_path, "vision_smoke_complete", bits=args.bits, cases=len(smoke["cases"]))
    print(json.dumps({"model": config["id"], "smoke": smoke}, ensure_ascii=False, indent=2), flush=True)
    if args.smoke_only:
        log_event(log_path, "smoke_only_complete", bits=args.bits)
        return 0

    latency = _latency(model, samples[0], log_path)
    log_event(log_path, "latency_complete", bits=args.bits, measurements=40)
    measured = evaluate(model, samples, log_path)
    measured["timing"]["warm_single_image"] = latency
    model_info = {
        "id": config["id"],
        "revision": config["revision"],
        "snapshot": str(snapshot),
        "bits": args.bits,
        "backbone": "9B Qwen3.5 vision-language model with Clef joint-schema head",
        "backend": "mlx-vlm vision path",
        "loader_sha256": sha256(snapshot / "clef_mlx.py"),
        "vision_enabled": bool(model.vision),
        "input_truncation": False,
        "max_length_tokens": 16384,
    }
    matched_count = inventory["matched_rating_count"]
    result = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "dataset": {
            "name": "JAFFE",
            "record": "https://zenodo.org/records/14974867",
            "doi": "10.5281/zenodo.14974867",
            "archive_bytes": archive_path.stat().st_size,
            "archive_md5": archive_md5,
            "readme_bytes": len(readme_bytes),
            "readme_md5": hashlib.md5(readme_bytes).hexdigest(),
            "image_count": len(samples),
            "source_image_mode": "8-bit grayscale TIFF; palette images verified grayscale then converted L→RGB for Qwen3-VL processor",
            "primary_rating_table_rows": inventory["rating_row_count"],
            "rating_join_matched_images": matched_count,
            "image_without_rating_count": len(inventory["image_without_rating_ids"]),
            "rating_rows_without_image_count": len(inventory["rating_without_image_ids"]),
            "fear_excluded_table_used": False,
        },
        "model": model_info,
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") if hasattr(os, "sysconf") else None,
            "default_mlx_device": str(mx.default_device()),
            "metal_available": bool(mx.metal.is_available()),
            "packages": _runtime_versions(),
        },
        "method": {
            "questions_per_image": 7,
            "requests_per_image": 1,
            "input_state": "fixed Japanese text; no filename, person code, pose code, or reference rating",
            "predicted_class_target": "requested posed-expression label in the filename; not a claim about felt emotion",
            "human_rating_target": "six README_FIRST full-table perceptual mean ratings (60 Japanese female observers as documented there)",
            "stored_per_image_predictions": False,
            "stored_image_payloads": False,
            "warmup_count": 5,
            "latency_measurement_count": 40,
        },
        "smoke": smoke,
        **measured,
        "provenance": {
            "rating_join_mismatches_preserved_without_correction": True,
            "image_without_rating_count": len(inventory["image_without_rating_ids"]),
            "rating_rows_without_image_count": len(inventory["rating_without_image_ids"]),
            "rating_metrics_denominator": matched_count,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    log_event(log_path, "run_complete", bits=args.bits, output=str(output_path), errors=measured["errors"])
    print(json.dumps({
        "output": str(output_path),
        "classification_accuracy": measured["classification"]["accuracy"],
        "classification_macro_f1": measured["classification"]["macro_f1"],
        "classification_errors": measured["classification"]["errors"],
        "rating_join_n": matched_count,
        "request_errors": measured["timing"]["full_suite"]["request_errors"],
        "suite_seconds": measured["timing"]["full_suite"]["total_seconds"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

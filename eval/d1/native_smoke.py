"""Native smoke for LiquidAI/d1-3B; exercises system_one and system_one_batch."""
from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

from server import D1Runtime, OFFICIAL_MODEL, REVISION


def checked_noul(answer: dict[str, Any], expected: bool) -> dict[str, Any]:
    p_yes = float(answer["noul"])
    if not math.isfinite(p_yes) or not 0 <= p_yes <= 1:
        raise AssertionError(f"invalid official noul P(yes): {p_yes}")
    return {"p_yes": p_yes, "expected_yes": expected, "threshold_match": (p_yes >= 0.5) == expected}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("eval/d1/native_smoke.json"))
    args = parser.parse_args()

    started = time.perf_counter()
    runtime = D1Runtime.load()
    load_seconds = time.perf_counter() - started
    if runtime.device != "mps" or runtime.dtype != "torch.bfloat16":
        raise AssertionError(f"unexpected selected execution: {runtime.device} {runtime.dtype}")
    torch.mps.synchronize()

    single_questions = {
        "language": {"type": "noul", "instructions": "この状態には日本語の文章が含まれていますか？"},
        "category": {
            "type": "choice", "instructions": "問い合わせの主な種類はどれですか？",
            "criteria": {"返品・返金": "購入商品の返品または返金", "配送": "配送状況や遅延"},
        },
        "satisfaction": {
            "type": "score", "instructions": "この感想の顧客満足度はどの程度ですか？",
            "criteria": ["非常に不満", "不満", "普通", "満足", "非常に満足"],
        },
    }
    single_started = time.perf_counter()
    single = runtime.model.system_one(
        {"text": "商品が壊れていたため、返品して返金してほしいです。", "状況": "購入済み商品"},
        single_questions,
    )
    torch.mps.synchronize()
    single_ms = (time.perf_counter() - single_started) * 1000
    if single["usage"]["input_tokens"] <= 0 or single["usage"]["output_tokens"] != 0:
        raise AssertionError(f"unexpected native usage: {single['usage']}")
    answers = single["answers"]
    if not 0 <= answers["category"]["confidence"] <= 1:
        raise AssertionError("invalid choice confidence")
    if not math.isclose(sum(answers["category"]["probabilities"].values()), 1.0, abs_tol=1e-5):
        raise AssertionError("choice probabilities do not sum to one")
    if not 0 <= answers["satisfaction"]["score"] <= 4:
        raise AssertionError("score is outside its zero-based 0..4 range")
    if not math.isclose(sum(answers["satisfaction"]["probabilities"].values()), 1.0, abs_tol=1e-5):
        raise AssertionError("score probabilities do not sum to one")

    direct_toxicity = {"type": "noul", "instructions": "この文は攻撃的・侮辱的な内容を含んでいますか？"}
    negated_toxicity = {"type": "noul", "instructions": "この文は攻撃的でも侮辱的でもない、穏当な内容ですか？"}
    batch_requests = [
        ("お前みたいな無能は今すぐ辞めろ。", {"q": {"type": "noul", "instructions": "この文は日本語ですか？"}}),
        ("This is a simple English sentence.", {"q": {"type": "noul", "instructions": "この文は日本語ですか？"}}),
        ("死ね。二度と連絡してくるな。", {"q": direct_toxicity}),
        ("先日はご丁寧に対応していただき、ありがとうございました。", {"q": direct_toxicity}),
        ("先日はご丁寧に対応していただき、ありがとうございました。", {"q": negated_toxicity}),
        ("死ね。二度と連絡してくるな。", {"q": negated_toxicity}),
    ]
    batch_started = time.perf_counter()
    batch = runtime.model.system_one_batch(batch_requests)
    torch.mps.synchronize()
    batch_ms = (time.perf_counter() - batch_started) * 1000
    if len(batch) != len(batch_requests):
        raise AssertionError(f"native batch returned {len(batch)} of {len(batch_requests)} answers")
    for item in batch:
        if item["usage"]["input_tokens"] <= 0 or item["usage"]["output_tokens"] != 0:
            raise AssertionError(f"unexpected native batch usage: {item['usage']}")

    batch_cases = {
        "japanese_positive": checked_noul(batch[0]["answers"]["q"], True),
        "japanese_negative_english": checked_noul(batch[1]["answers"]["q"], False),
        "toxicity_positive": checked_noul(batch[2]["answers"]["q"], True),
        "toxicity_negative": checked_noul(batch[3]["answers"]["q"], False),
        "explicitly_negated_benign_positive": checked_noul(batch[4]["answers"]["q"], True),
        "explicitly_negated_hostile_negative": checked_noul(batch[5]["answers"]["q"], False),
    }
    report = {
        "official_model": OFFICIAL_MODEL,
        "revision": REVISION,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "transformers": __import__("transformers").__version__,
        "device": runtime.device,
        "dtype": runtime.dtype,
        "model_load_seconds": round(load_seconds, 3),
        "native_system_one": {
            "method": "model.system_one(state, named_questions)",
            "elapsed_ms": round(single_ms, 1),
            "usage": single["usage"],
            "answers": answers,
        },
        "native_system_one_batch": {
            "method": "model.system_one_batch(requests)",
            "request_count": len(batch_requests),
            "elapsed_ms": round(batch_ms, 1),
            "cases": batch_cases,
            "usage": [item["usage"] for item in batch],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

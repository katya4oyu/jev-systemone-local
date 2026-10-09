"""Exercise the official TypeSafe Python SDK against the task-local real-model adapter."""
from __future__ import annotations

import argparse
import json
import math
from urllib.parse import urlparse

from typesafe_sdk import Choice, Noul, Score, TypeSafeClient

from server import MODEL_ID


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8025")
    args = parser.parse_args()
    hostname = urlparse(args.base_url).hostname
    if hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("the SDK smoke test is restricted to loopback URLs")
    state = {
        "問い合わせ": "商品が壊れていたため、返品して返金してほしいです。",
        "状況": "配送ではなく、購入済み商品の不具合について相談しています。",
    }
    questions = {
        "japanese": Noul(instructions="この状態には日本語の文章が含まれていますか？"),
        "category": Choice(
            instructions="この問い合わせの主な種類はどれですか？",
            criteria={"返品・返金": None, "配送": None, "技術トラブル": None},
        ),
        "urgency": Score(
            instructions="この問い合わせの緊急度はどの程度ですか？",
            criteria=["急がない", "通常", "やや急ぐ", "急ぐ", "非常に急ぐ"],
        ),
    }
    # The SDK requires a non-empty API key during construction. This local-only
    # placeholder is ignored by this loopback adapter; it is not a credential.
    with TypeSafeClient(api_key="local-evaluation-no-auth", base_url=args.base_url,
                        model=MODEL_ID) as client:
        result = client.system_one(state, questions, model=MODEL_ID)
    if result.model != MODEL_ID:
        raise AssertionError(f"SDK-selected model mismatch: {result.model!r}")
    if result.usage.input_tokens <= 0 or result.usage.output_tokens != 0:
        raise AssertionError(f"unexpected usage: {result.usage}")

    noul = result.nouls["japanese"]
    choice = result.choices["category"]
    score = result.scores["urgency"]
    if not math.isfinite(noul.noul) or not 0 <= noul.noul <= 1:
        raise AssertionError(f"invalid noul P(yes): {noul.noul}")
    if choice.choice not in questions["category"].criteria:
        raise AssertionError(f"choice outside request options: {choice.choice!r}")
    if not math.isclose(sum(choice.probabilities.values()), 1.0, abs_tol=1e-5):
        raise AssertionError(f"choice probabilities do not sum to one: {choice.probabilities}")
    if not math.isfinite(score.score) or not 0 <= score.score <= 4:
        raise AssertionError(f"score outside zero-based 0..4 scale: {score.score}")
    if not math.isclose(sum(score.probabilities.values()), 1.0, abs_tol=1e-5):
        raise AssertionError(f"score probabilities do not sum to one: {score.probabilities}")

    print(json.dumps({
        "sdk": "typesafe-sdk",
        "model": result.model,
        "usage": result.usage.model_dump(),
        "answers": {
            "japanese": {"type": "noul", "p_yes": noul.noul},
            "category": {
                "type": "choice", "choice": choice.choice,
                "confidence": choice.confidence,
                "probabilities": choice.probabilities,
            },
            "urgency": {
                "type": "score", "score": score.score,
                "confidence": score.confidence,
                "probabilities": score.probabilities,
            },
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

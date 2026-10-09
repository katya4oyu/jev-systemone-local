"""Real-model HTTP contract smoke for the task-local d1-3B adapter."""
from __future__ import annotations

import argparse
import json
import math
from urllib.parse import urlparse
from typing import Any

import httpx

from server import MODEL_ID, OFFICIAL_MODEL, REVISION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8025")
    args = parser.parse_args()
    if urlparse(args.base_url).hostname not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("HTTP smoke is restricted to loopback URLs")

    payload = {
        "model": MODEL_ID,
        "state": {"text": "商品が壊れていたため、返品して返金してほしいです。"},
        "questions": {
            "language": {"type": "noul", "instructions": "この状態には日本語が含まれていますか？"},
            "category": {
                "type": "choice", "instructions": "問い合わせの種類は？",
                "criteria": {"返品・返金": "返品または返金の依頼", "配送": "配送状況の質問"},
            },
            "satisfaction": {
                "type": "score", "instructions": "顧客満足度は？",
                "criteria": ["非常に不満", "不満", "普通", "満足", "非常に満足"],
            },
        },
    }
    with httpx.Client(base_url=args.base_url, timeout=120) as client:
        health = client.get("/healthz")
        health.raise_for_status()
        identity = health.json()["model"]
        if (identity["id"], identity["official_model"], identity["revision"],
                identity["device"], identity["dtype"]) != (
                MODEL_ID, OFFICIAL_MODEL, REVISION, "mps", "torch.bfloat16"):
            raise AssertionError(f"health identity mismatch: {identity}")
        listed = client.get("/v1/models")
        listed.raise_for_status()
        model_names = [item["name"] for item in listed.json()["models"]]
        if model_names != [MODEL_ID]:
            raise AssertionError(f"unexpected model list: {model_names}")

        response = client.post("/v1/systemone", json=payload)
        response.raise_for_status()
        result = response.json()
        if result["model"] != MODEL_ID or result["usage"]["output_tokens"] != 0:
            raise AssertionError(f"unexpected model/usage: {result.get('model')} {result.get('usage')}")
        if result["usage"]["input_tokens"] <= 0:
            raise AssertionError(f"invalid input usage: {result['usage']}")
        answers: dict[str, Any] = result["answers"]
        if {key: answer["type"] for key, answer in answers.items()} != {
            "language": "noul", "category": "choice", "satisfaction": "score",
        }:
            raise AssertionError(f"answer type mismatch: {answers}")
        for key in ("category", "satisfaction"):
            probs = answers[key]["probabilities"].values()
            if not math.isclose(sum(probs), 1.0, abs_tol=1e-5):
                raise AssertionError(f"probabilities do not sum to one for {key}")
        mismatch = client.post("/v1/systemone", json={**payload, "model": "not-loaded"})
        if mismatch.status_code != 422:
            raise AssertionError(f"wrong model was not rejected: {mismatch.status_code}")

    print(json.dumps({
        "health": identity,
        "listed_model_names": model_names,
        "answer": result,
        "wrong_model_http_status": mismatch.status_code,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

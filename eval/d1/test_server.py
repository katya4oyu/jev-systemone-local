from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from server import MODEL_ID, create_app


class FakeRuntime:
    max_input_tokens = 8

    def __init__(self) -> None:
        self.calls = 0

    def public_identity(self) -> dict[str, Any]:
        return {
            "id": MODEL_ID,
            "official_model": "LiquidAI/d1-3B",
            "revision": "test-revision",
            "device": "mps",
            "dtype": "torch.bfloat16",
            "max_input_tokens": self.max_input_tokens,
        }

    def prepared_input_tokens(self, state: Any, questions: dict[str, Any]) -> int:
        return len(state)

    def system_one(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        answers = {}
        for name, question in questions.items():
            kind = question["type"]
            if kind == "noul":
                answers[name] = {"type": "noul", "noul": 0.8}
            elif kind == "choice":
                labels = list(question["criteria"])
                answers[name] = {
                    "type": "choice", "choice": labels[0], "confidence": 0.75,
                    "probabilities": {label: 0.75 if i == 0 else 0.25 / (len(labels) - 1)
                                      for i, label in enumerate(labels)},
                }
            else:
                answers[name] = {
                    "type": "score", "score": 1.25, "confidence": 0.6,
                    "probabilities": {"0": 0.4, "1": 0.3, "2": 0.3},
                    "legend": {"0": "low", "1": "mid", "2": "high"},
                }
        return {"answers": answers, "usage": {"input_tokens": 7, "output_tokens": 0}}


def test_health_and_models_report_loaded_adapter_identity() -> None:
    runtime = FakeRuntime()
    with TestClient(create_app(runtime)) as client:
        assert client.get("/health").json()["model"]["id"] == MODEL_ID
        models = client.get("/v1/models").json()["models"]
        assert [model["name"] for model in models] == [MODEL_ID]
        assert "not a vendor server" in models[0]["description"]


def test_specialized_answers_are_returned_with_selected_model() -> None:
    runtime = FakeRuntime()
    payload = {
        "model": MODEL_ID,
        "state": "こんにちは",
        "questions": {
            "n": {"type": "noul", "instructions": "日本語の文章ですか？"},
            "c": {"type": "choice", "instructions": "分類は？", "criteria": {"a": "A", "b": "B"}},
            "s": {"type": "score", "instructions": "評価は？", "criteria": ["low", "mid", "high"]},
        },
    }
    with TestClient(create_app(runtime)) as client:
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 200
        result = response.json()
        assert result["model"] == MODEL_ID
        assert result["usage"]["output_tokens"] == 0
        assert result["answers"]["n"]["noul"] == 0.8
        assert result["answers"]["c"]["choice"] == "a"
        assert result["answers"]["s"]["score"] == 1.25
        assert runtime.calls == 1
        stats = client.get("/_task/stats").json()
        assert stats["successful_requests"] == 1
        assert stats["questions"] == 3
        assert stats["question_types"] == {"choice": 1, "noul": 1, "score": 1}
        assert stats["max_prepared_input_tokens"] == len("こんにちは")


def test_model_must_be_explicit_and_match_loaded_checkpoint() -> None:
    runtime = FakeRuntime()
    payload = {"model": "some-other-model", "state": "a", "questions": {
        "q": {"type": "noul", "instructions": "yes?"},
    }}
    with TestClient(create_app(runtime)) as client:
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 422
        assert runtime.calls == 0
        assert client.get("/_task/stats").json()["rejected_requests"] == 1


def test_exact_prepared_input_limit_is_allowed_and_one_over_is_rejected() -> None:
    runtime = FakeRuntime()
    payload = {"model": MODEL_ID, "state": "12345678", "questions": {
        "q": {"type": "noul", "instructions": "within bound"},
    }}
    with TestClient(create_app(runtime)) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 200
        payload["state"] = "123456789"
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 422
        assert "9 tokens exceeds the context window of 8 tokens" in response.text
        assert runtime.calls == 1
        stats = client.get("/_task/stats").json()
        assert stats["successful_requests"] == 1
        assert stats["rejected_requests"] == 1

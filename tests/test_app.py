from fastapi.testclient import TestClient

from jev_systemone_local.app import create_app
from jev_systemone_local.laya_mlx_backend import InputTooLong


class FakeBackend:
    backend_name = "laya-mlx"
    model_name = "laya-multilingual-mlx"
    checkpoint = "test-only"

    def evaluate(self, state, questions):
        if state == "too long":
            raise InputTooLong("state exceeds model context")
        assert questions["yes"]["type"] == "noul"
        return {"model": self.model_name, "answers": {"yes": {"type": "noul", "noul": 0.8}},
                "usage": {"input_tokens": 8, "output_tokens": 0}}


def request(state="short", model="jev-latest"):
    return {"model": model, "state": state,
            "questions": {"yes": {"type": "noul", "instructions": "Is this urgent?"}}}


def test_playground_and_decision():
    with TestClient(create_app(FakeBackend())) as client:
        assert "判断する" in client.get("/").text
        assert client.get("/healthz").json()["status"] == "ready"
        assert client.get("/v1/models").json()["models"][0]["backend"] == "laya-mlx"
        result = client.post("/v1/systemone", json=request())
        assert result.status_code == 200
        assert result.json() == {"model": "laya-multilingual-mlx",
                                 "answers": {"yes": {"type": "noul", "noul": 0.8}},
                                 "usage": {"input_tokens": 8, "output_tokens": 0}}


def test_validation_and_overlength_are_explicit():
    with TestClient(create_app(FakeBackend())) as client:
        assert client.post("/v1/systemone", json=request(model="unknown")).status_code == 422
        assert client.post("/v1/systemone", json=request(state="too long")).status_code == 422
        malformed = request()
        malformed["questions"]["yes"]["type"] = "choice"
        assert client.post("/v1/systemone", json=malformed).status_code == 422

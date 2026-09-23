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


def test_playground_uses_served_official_sdk():
    with TestClient(create_app(FakeBackend())) as client:
        page = client.get("/").text
        assert '<script type="module">' in page
        assert 'from "/vendor/typesafe-sdk.mjs"' in page
        assert "baseURL:window.location.origin" in page
        assert "dangerouslyAllowBrowser:true" in page
        assert "client.systemOne(payload)" in page
        assert "client.models.list()" in page
        sdk = client.get("/vendor/typesafe-sdk.mjs")
        assert sdk.status_code == 200
        assert sdk.headers["content-type"].startswith("text/javascript")
        assert 'const VERSION = "0.6.0"' in sdk.text


def test_validation_and_overlength_are_explicit():
    with TestClient(create_app(FakeBackend())) as client:
        assert client.post("/v1/systemone", json=request(model="unknown")).status_code == 422
        assert client.post("/v1/systemone", json=request(state="too long")).status_code == 422
        malformed = request()
        malformed["questions"]["yes"]["type"] = "choice"
        assert client.post("/v1/systemone", json=malformed).status_code == 422


def test_model_name_routes_both_engines_without_changing_legacy_alias():
    class OtherBackend(FakeBackend):
        backend_name = "laya-coreml"
        model_name = "laya-multilingual-coreml"

    class ShortBackend(OtherBackend):
        model_name = "laya-multilingual-coreml-ane"

        def evaluate(self, state, questions):
            if state == "too long":
                raise InputTooLong("96 token limit")
            return super().evaluate(state, questions)

    backends = {b.model_name: b for b in (FakeBackend(), OtherBackend(), ShortBackend())}
    with TestClient(create_app(backends=backends)) as client:
        listed = client.get("/v1/models").json()["models"]
        assert {m["name"] for m in listed} == {"jev-latest", *backends}
        assert client.post("/v1/systemone", json=request()).json()["model"] == "laya-multilingual-mlx"
        for model in ("laya-multilingual-coreml", "laya-multilingual-coreml-ane"):
            answer = client.post("/v1/systemone", json=request(model=model))
            assert answer.status_code == 200
            assert answer.json()["model"] == model
        assert client.post("/v1/systemone", json=request(model="no-such-model")).status_code == 422
        too_long = client.post("/v1/systemone", json=request("too long", ShortBackend.model_name))
        assert too_long.status_code == 422
        assert "96 token" in too_long.json()["detail"]

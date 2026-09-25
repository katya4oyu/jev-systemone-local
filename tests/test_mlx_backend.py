from types import SimpleNamespace

import laya_mlx
import pytest
from fastapi.testclient import TestClient
from laya_mlx.agent import Agent

from jev_systemone_local.app import create_app
from jev_systemone_local.laya_mlx_backend import LayaMLXBackend


class Tokens:
    mask_token = "[MASK]"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [4] * len(text.split())}


def make_backend(monkeypatch, encoder_limit=8192):
    agent = SimpleNamespace(
        tok=Tokens(), cfg={"max_len": 1024, "head_max_len": 256},
        encoder_cfg={"max_position_embeddings": encoder_limit},
        _to_internal=Agent._to_internal,
    )
    agent.predict = lambda state, questions: {
        "answers": {"yes": {"noul": 0.7}},
        "usage": {"input_tokens": len(state.split()), "output_tokens": 0},
    }
    monkeypatch.setattr(laya_mlx, "load", lambda checkpoint: agent)
    return LayaMLXBackend()


def test_mlx_uses_encoder_context_without_changing_question_budget(monkeypatch):
    backend = make_backend(monkeypatch)
    assert backend.agent.cfg == {"max_len": 8192, "head_max_len": 256}


def test_mlx_respects_smaller_encoder_limit(monkeypatch):
    assert make_backend(monkeypatch, 2048).agent.cfg["max_len"] == 2048


@pytest.mark.parametrize("model", ["jev-latest", "laya-multilingual-mlx"])
def test_mlx_http_accepts_exact_context_and_rejects_overflow(monkeypatch, model):
    backend = make_backend(monkeypatch)
    question = {"type": "noul", "instructions": "Is it urgent?"}
    prefix, _ = backend.common.build_prefix(
        backend.agent.tok, backend.agent._to_internal(question), 256,
    )
    available = 8192 - len(prefix) - 1
    with TestClient(create_app(backend, snake_policies={})) as client:
        for length, status in [(8, 200), (1500, 200), (available, 200), (available + 1, 422)]:
            response = client.post("/v1/systemone", json={
                "model": model, "state": "word " * length, "questions": {"yes": question},
            })
            assert response.status_code == status
            if status == 200:
                assert response.json()["model"] == "laya-multilingual-mlx"
            else:
                assert "only" in response.json()["detail"]

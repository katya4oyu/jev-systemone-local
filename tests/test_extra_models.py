from types import SimpleNamespace

import laya_mlx
import pytest
from fastapi.testclient import TestClient
from laya_mlx.agent import Agent

from systemone_workbench.app import create_app
from systemone_workbench.extra_models import load_extra_backends, parse_spec
from systemone_workbench.laya_mlx_backend import LayaMLXBackend
from test_app import FakeBackend, request
from test_mlx_backend import Tokens


def test_parse_spec_accepts_names_and_paths():
    assert parse_spec("", set()) == {}
    assert parse_spec(" ja=/models/ja-mlx , en=org/repo ", set()) == {"ja": "/models/ja-mlx", "en": "org/repo"}


@pytest.mark.parametrize("spec,taken", [
    ("no-equals", set()), ("=path", set()), ("name=", set()),
    ("a=x,a=y", set()), ("jev-latest=x", set()), ("laya-multilingual-mlx=x", {"laya-multilingual-mlx"}),
])
def test_parse_spec_rejects_ambiguous_or_shadowing_entries(spec, taken):
    with pytest.raises(ValueError):
        parse_spec(spec, taken)


def test_load_extra_backends_passes_name_and_keeps_trained_context():
    calls = []

    def factory(checkpoint, **kwargs):
        calls.append((checkpoint, kwargs))
        return SimpleNamespace(model_name=kwargs["model_name"])

    loaded = load_extra_backends({"base"}, factory, spec="ja=/m/ja")
    assert set(loaded) == {"ja"}
    assert calls == [("/m/ja", {"model_name": "ja", "extend_context": False})]


def test_load_failure_is_not_swallowed():
    def factory(checkpoint, **kwargs):
        raise FileNotFoundError(checkpoint)

    with pytest.raises(FileNotFoundError):
        load_extra_backends(set(), factory, spec="ja=/missing")


def make_backend(monkeypatch, **kwargs):
    agent = SimpleNamespace(
        tok=Tokens(), cfg={"max_len": 1024, "head_max_len": 256},
        encoder_cfg={"max_position_embeddings": 8192}, _to_internal=Agent._to_internal)
    agent.predict = lambda state, questions: {
        "answers": {"yes": {"noul": 0.7}}, "usage": {"input_tokens": len(state.split()), "output_tokens": 0}}
    monkeypatch.setattr(laya_mlx, "load", lambda checkpoint: agent)
    return LayaMLXBackend(**kwargs)


def test_fine_tuned_backend_keeps_its_trained_length_and_reports_its_name(monkeypatch, tmp_path):
    backend = make_backend(monkeypatch, checkpoint=str(tmp_path), model_name="laya-ja-turn-mlx", extend_context=False)
    assert backend.agent.cfg["max_len"] == 1024
    assert backend.model_name == "laya-ja-turn-mlx"
    assert backend.checkpoint == f"local:{tmp_path.name}"  # no absolute path leaks to /v1/models
    assert make_backend(monkeypatch).agent.cfg["max_len"] == 8192  # built-in default unchanged


def test_request_model_selects_the_extra_backend():
    class Other(FakeBackend):
        model_name = "laya-ja-turn-mlx"

    backends = {"laya-multilingual-mlx": FakeBackend(), "laya-ja-turn-mlx": Other()}
    with TestClient(create_app(backends=backends)) as client:
        assert client.post("/v1/systemone", json=request(model="laya-ja-turn-mlx")).json()["model"] == "laya-ja-turn-mlx"
        assert client.post("/v1/systemone", json=request(model="jev-latest")).json()["model"] == "laya-multilingual-mlx"
        names = [m["name"] for m in client.get("/v1/models").json()["models"]]
        assert "laya-ja-turn-mlx" in names


def test_former_env_names_are_still_read_and_conflicts_are_rejected(monkeypatch):
    from systemone_workbench.extra_models import LEGACY_ENV_VAR, LEGACY_PROXY_ENV_VAR, read_env

    monkeypatch.delenv("SYSTEMONE_MLX_MODELS", raising=False)
    monkeypatch.delenv("SYSTEMONE_PROXY_MODELS", raising=False)
    monkeypatch.setenv(LEGACY_ENV_VAR, "ja=/m/ja")
    monkeypatch.setenv(LEGACY_PROXY_ENV_VAR, "kev-4b=http://127.0.0.1:8009")
    assert read_env("SYSTEMONE_MLX_MODELS", LEGACY_ENV_VAR) == "ja=/m/ja"
    assert list(load_extra_backends(set(), lambda c, **kw: c)) == ["ja"]  # falls back to the former name
    monkeypatch.setenv("SYSTEMONE_MLX_MODELS", "ja=/m/ja")  # same value under both names is fine
    assert read_env("SYSTEMONE_MLX_MODELS", LEGACY_ENV_VAR) == "ja=/m/ja"
    monkeypatch.setenv("SYSTEMONE_MLX_MODELS", "en=/m/en")
    with pytest.raises(ValueError, match="both set"):
        read_env("SYSTEMONE_MLX_MODELS", LEGACY_ENV_VAR)

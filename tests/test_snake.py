"""The HTTP game must retain real per-step decisions and isolated sessions."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from laya_mlx.snake.game import DIRECTIONS
from laya_mlx.snake.policy import Decision

from jev_systemone_local.app import create_app
from jev_systemone_local.snake_demo import SnakeService, policy_for_backend


class FakePolicy:
    guarded = True

    def __init__(self):
        self.calls = 0

    def decide(self, game):
        self.calls += 1
        safe = next(move for move in game.moves() if move.safe)
        proposed = next(move.direction for move in game.moves() if not move.safe)
        return Decision(
            probabilities={key: 0.7 if key == proposed else 0.1 for key in DIRECTIONS},
            proposed=proposed, executed=safe.direction,
            safe_directions=[m.direction for m in game.moves() if m.safe],
            intervened=True, dead_end_risk=0.25, food_reachable=0.9,
            inference_ms=3.2, decision_ms=4.1,
            input_tokens=30, output_tokens=0, safe_count=1, planner_best=safe.direction,
        )


class FakeBackend:
    backend_name = "laya-mlx"
    model_name = "laya-multilingual-mlx"


def test_snake_page_and_separate_real_step_contract():
    with TestClient(create_app(FakeBackend(), snake_policy=FakePolicy())) as client:
        page = client.get("/snake")
        assert page.status_code == 200
        assert "直前の判断" in page.text
        assert "'/snake/api/sessions'" in page.text
        first = client.post("/snake/api/sessions").json()
        second = client.post("/snake/api/sessions").json()
        assert first["session"] != second["session"]
        assert first["guarded"] is True
        assert first["board"]["width"] == 24
        moved = client.post(f"/snake/api/sessions/{first['session']}/step")
        assert moved.status_code == 200
        data = moved.json()
        assert data["previous_board"] == first["board"]
        assert data["board"]["ticks"] == 1
        assert data["decision"]["intervened"] is True
        assert data["decision"]["executed"] in data["decision"]["safe_directions"]
        assert data["interventions"] == 1
        assert client.post(f"/snake/api/sessions/{second['session']}/step").json()["board"]["ticks"] == 1
        assert client.post(f"/snake/api/sessions/{first['session']}/step").json()["board"]["ticks"] == 2
        assert client.post("/snake/api/sessions/not-a-game/step").status_code == 404


def test_snake_session_uses_selected_model_for_every_step():
    mlx = FakePolicy()
    coreml = FakePolicy()
    snake = SnakeService(mlx, policies={"laya-multilingual-mlx": mlx,
                                        "laya-multilingual-coreml": coreml})
    session = snake.start(seed=1, model="laya-multilingual-coreml")
    assert session["model"] == "laya-multilingual-coreml"
    snake.step(session["session"])
    assert coreml.calls == 1
    assert mlx.calls == 0
    with pytest.raises(ValueError, match="unsupported model"):
        snake.start(model="not-installed")


def test_snake_http_start_selects_model_without_changing_legacy_request():
    mlx = FakePolicy()
    coreml = FakePolicy()
    with TestClient(create_app(
        backends={"laya-multilingual-mlx": FakeBackend()},
        snake_policies={"laya-multilingual-mlx": mlx,
                        "laya-multilingual-coreml": coreml},
    )) as client:
        old = client.post("/snake/api/sessions")
        assert old.status_code == 200
        assert old.json()["model"] == "laya-multilingual-mlx"
        new = client.post("/snake/api/sessions", json={"model": "laya-multilingual-coreml"})
        assert new.status_code == 200
        client.post(f"/snake/api/sessions/{new.json()['session']}/step")
        assert coreml.calls == 1 and mlx.calls == 0
        assert client.post("/snake/api/sessions", json={"model": "unknown"}).status_code == 422


def test_limited_sessions_and_shared_loaded_agent():
    service = SnakeService(FakePolicy())
    first = service.start(seed=7)["session"]
    for _ in range(service.max_sessions):
        service.start()
    assert first not in service.sessions
    assert len(service.sessions) == service.max_sessions
    agent = object()
    policy = policy_for_backend(SimpleNamespace(agent=agent))
    assert policy.agent is agent
    assert policy.guarded and policy.prompt == "compact"

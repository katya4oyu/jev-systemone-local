"""One real Laya inference per Snake move, with per-browser game sessions."""
from __future__ import annotations

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from laya_mlx.snake.game import SnakeGame
from laya_mlx.snake.policy import LayaPolicy


class UnknownGame(LookupError):
    pass


class FinishedGame(RuntimeError):
    pass


@dataclass
class Session:
    game: SnakeGame
    touched: float
    model: str = "laya-multilingual-mlx"
    interventions: int = 0


class SnakeService:
    """In-memory, bounded games; caller serializes model and game access."""

    max_sessions = 16
    idle_seconds = 3600

    def __init__(self, policy: Any, policies: dict[str, Any] | None = None):
        self.policy = policy
        self.policies = policies if policies is not None else {"laya-multilingual-mlx": policy}
        self.sessions: OrderedDict[str, Session] = OrderedDict()

    def _prune(self):
        now = time.monotonic()
        for key, session in list(self.sessions.items()):
            if now - session.touched > self.idle_seconds:
                del self.sessions[key]


    def start(self, seed: int | None = None, model: str = "laya-multilingual-mlx") -> dict:
        self._prune()
        if model not in self.policies:
            raise ValueError(f"unsupported model: {model}")
        while len(self.sessions) >= self.max_sessions:
            self.sessions.popitem(last=False)
        session_id = secrets.token_urlsafe(18)
        game = SnakeGame(seed=secrets.randbelow(2**31) if seed is None else seed)
        self.sessions[session_id] = Session(game, time.monotonic(), model)
        return {"session": session_id, "board": game.snapshot(),
                "guarded": self.policies[model].guarded, "model": model}

    def step(self, session_id: str) -> dict:
        self._prune()
        session = self.sessions.get(session_id)
        if session is None:
            raise UnknownGame("Game not found or expired; start a new round")
        game = session.game
        if not game.alive or game.won:
            raise FinishedGame("Round already finished; start a new round")
        before = game.snapshot()
        decision = self.policies[session.model].decide(game)
        if decision.executed not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError("Model returned an invalid move")
        game.step(decision.executed)
        session.interventions += int(decision.intervened)
        session.touched = time.monotonic()
        self.sessions.move_to_end(session_id)
        return {"previous_board": before, "board": game.snapshot(),
                "decision": decision.to_dict(), "interventions": session.interventions}


def policy_for_backend(backend: Any) -> Any:
    """Reuse the HTTP backend's already-loaded Agent, not a second weights copy.

    LayaPolicy.decide only reads agent/guarded/prompt. Its constructor is
    intentionally skipped because that would load a second checkpoint.
    """
    if getattr(backend, "backend_name", None) == "laya-coreml":
        from laya_coreml.snake.policy import LayaPolicy as CoreMLPolicy
        policy_class = CoreMLPolicy
    else:
        policy_class = LayaPolicy
    policy = object.__new__(policy_class)
    policy.agent = backend.agent
    policy.guarded = True
    policy.prompt = "compact"
    return policy

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
    interventions: int = 0


class SnakeService:
    """In-memory, bounded games; caller serializes model and game access."""

    max_sessions = 16
    idle_seconds = 3600

    def __init__(self, policy: Any):
        self.policy = policy
        self.sessions: OrderedDict[str, Session] = OrderedDict()

    def _prune(self):
        now = time.monotonic()
        for key, session in list(self.sessions.items()):
            if now - session.touched > self.idle_seconds:
                del self.sessions[key]


    def start(self, seed: int | None = None) -> dict:
        self._prune()
        while len(self.sessions) >= self.max_sessions:
            self.sessions.popitem(last=False)
        session_id = secrets.token_urlsafe(18)
        game = SnakeGame(seed=secrets.randbelow(2**31) if seed is None else seed)
        self.sessions[session_id] = Session(game, time.monotonic())
        return {"session": session_id, "board": game.snapshot(), "guarded": self.policy.guarded}

    def step(self, session_id: str) -> dict:
        self._prune()
        session = self.sessions.get(session_id)
        if session is None:
            raise UnknownGame("Game not found or expired; start a new round")
        game = session.game
        if not game.alive or game.won:
            raise FinishedGame("Round already finished; start a new round")
        before = game.snapshot()
        decision = self.policy.decide(game)
        if decision.executed not in {"UP", "DOWN", "LEFT", "RIGHT"}:
            raise ValueError("Model returned an invalid move")
        game.step(decision.executed)
        session.interventions += int(decision.intervened)
        session.touched = time.monotonic()
        self.sessions.move_to_end(session_id)
        return {"previous_board": before, "board": game.snapshot(),
                "decision": decision.to_dict(), "interventions": session.interventions}


def policy_for_backend(backend: Any) -> LayaPolicy:
    """Reuse the HTTP backend's already-loaded MLX Agent, not a second weights copy.

    LayaPolicy.decide only reads agent/guarded/prompt. Its constructor is
    intentionally skipped because that would load a second checkpoint.
    """
    policy = object.__new__(LayaPolicy)
    policy.agent = backend.agent
    policy.guarded = True
    policy.prompt = "compact"
    return policy

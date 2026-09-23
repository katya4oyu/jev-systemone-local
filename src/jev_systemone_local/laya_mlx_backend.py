"""The Laya-MLX implementation behind the System One wire contract."""

from __future__ import annotations

import json
import math
from typing import Any


class InputTooLong(ValueError):
    """The complete state cannot fit in a model question's context window."""


class LayaMLXBackend:
    backend_name = "laya-mlx"
    model_name = "laya-multilingual-mlx"

    def __init__(self, checkpoint: str = "aac6fef/laya-multilingual-mlx") -> None:
        import laya_mlx

        self.checkpoint = checkpoint
        self.agent = laya_mlx.load(checkpoint)

    def evaluate(self, state: str | dict | list, questions: dict[str, dict]) -> dict[str, Any]:
        # Laya's build_sequence truncates state without alerting its caller. A decision
        # over an incomplete state would be misleading, so reject that input first.
        from laya_mlx.common import build_prefix, serialize_state

        agent = self.agent
        max_len = agent.cfg.get("max_len", 512)
        head_max_len = agent.cfg.get("head_max_len", 192)
        text = serialize_state(state).replace(agent.tok.mask_token, " ")
        state_ids = agent.tok(text, add_special_tokens=False)["input_ids"]
        for name, question in questions.items():
            q = agent._to_internal(question)
            prefix, markers = build_prefix(agent.tok, q, head_max_len)
            if len(markers) != (len(question["criteria"]) if q["t"] != "noul" else 2):
                raise InputTooLong(f"Question {name!r} exceeds the options budget")
            available = max_len - len(prefix) - 1
            if available < len(state_ids):
                raise InputTooLong(
                    f"Question {name!r} needs {len(state_ids)} state tokens, "
                    f"but only {max(0, available)} fit; shorten the state or criteria"
                )
        result = agent.predict(state, questions)
        answers: dict[str, dict] = {}
        for name, question in questions.items():
            raw = result["answers"][name]
            kind = question["type"]
            if kind == "noul":
                answers[name] = {"type": kind, "noul": raw["noul"]}
            elif kind == "choice":
                answers[name] = {
                    "type": kind,
                    "choice": raw["choice"],
                    "probabilities": raw["probabilities"],
                    "confidence": raw["confidence"],
                }
            else:
                answers[name] = {
                    "type": kind,
                    "score": raw["score"],
                    "legend": {str(i): json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
                               for i, v in enumerate(question["criteria"])},
                    "probabilities": raw["probabilities"],
                    "confidence": raw["confidence"],
                }
            for value in answers[name].values():
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Non-finite inference result")
        return {"model": self.model_name, "answers": answers, "usage": result["usage"]}

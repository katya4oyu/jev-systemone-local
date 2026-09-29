"""Client-compatible wrapper that runs a (fine-tuned) Laya checkpoint in-process with PyTorch.

Lets every eval/ja script run against a local checkpoint directory via `main(LocalClient(dir))`.
"""

from __future__ import annotations

import time

import laya


class LocalClient:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.agent = laya.load(checkpoint, device=device) if device else laya.load(checkpoint)

    def ask(self, model, state, questions):
        t = time.perf_counter()
        try:
            resp = self.agent.predict(state, questions)
        except Exception as exc:  # over-long input etc.
            return None, (time.perf_counter() - t) * 1000, f"local {exc}"[:120]
        return resp, (time.perf_counter() - t) * 1000, ""

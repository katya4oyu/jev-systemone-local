"""Evaluate a Laya checkpoint (base or fine-tuned) in-process on the held-out Japanese eval sets.

Runs: dialog tasks (EOU / react / memory / drift, 2 forms), unseen wordings, and regression checks on the
earlier tasks (ticket classification with 2..50 labels, toxicity noul incl. negation) plus single-question latency.
Results go to eval/ja/results_*_<tag>.json.

Usage: python eval_ft.py <checkpoint_dir_or_hf_id> <tag>
"""

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

ckpt, tag = sys.argv[1], sys.argv[2]
os.environ["EVAL_TAG"] = tag
os.environ.setdefault("EVAL_MODEL", "local")  # makes run.py evaluate a single model entry
sys.path[:0] = [str(Path(__file__).parent), str(Path(__file__).parent.parent / "ja")]

import json  # noqa: E402

import run  # noqa: E402
import run_dialog  # noqa: E402
import run_dialog_wordings  # noqa: E402
from ft_client import LocalClient  # noqa: E402

c = LocalClient(ckpt)
run_dialog.main(c)
run_dialog_wordings.main(c)

out = {}
run.run_choice(c, out)   # ticket classification, 2/5/10/20/50 labels (regression check)
run.run_noul(c, out)     # toxicity noul incl. negated phrasing (regression check)
for _ in range(5):
    c.ask("x", "商品が壊れていました。", {"q": {"type": "noul", "instructions": "この文は苦情ですか？"}})
ts = sorted(c.ask("x", "商品が壊れていました。", {"q": {"type": "noul", "instructions": "この文は苦情ですか？"}})[1] for _ in range(40))
out["latency_single_ms"] = {"p50": round(statistics.median(ts), 1), "p95": round(ts[int(len(ts) * .95)], 1),
                            "note": "PyTorch on MPS in-process, not the MLX server"}
Path(__file__).parent.parent.joinpath("ja", run.result_name("results_regression.json")).write_text(
    json.dumps(out, ensure_ascii=False, indent=1))
for k, v in next(iter(out.get("choice_scaling", {}).values()), {}).items():
    print("ticket", k, "labels acc", v["acc"], "ece", v["ece"])
print("noul", {k: v["acc"] for k, v in out["noul"].items()}, "latency", out["latency_single_ms"])

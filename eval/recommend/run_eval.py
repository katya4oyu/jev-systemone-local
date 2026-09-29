"""Accuracy of /v1/recommend's task and constraint detection on the held-out eval/recommend/requests_ja.py.

Usage: uv run python eval/recommend/run_eval.py
No server needed (the classifier is pure Python). The classifier is trained only on train_requests_{a,b,c}.py.
The first 6 requests per task are the dev split (looked at while tuning the constraint patterns), the last 6 the
untouched test split.
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from requests_ja import REQUESTS  # noqa: E402

from jev_systemone_local import recommend as R  # noqa: E402

seen = defaultdict(int)
rows = []
for item in REQUESTS:
    seen[item["task"]] += 1
    det = R.detect_task(item["text"])
    final = R.recommend(item["text"], set())["task"]["id"]  # after the stated-label-count rule
    rows.append({"final": final, "split": "dev" if seen[item["task"]] <= 6 else "test", "gold": item["task"], "pred": det["task"],
                 "conf": det["confidence"], "flags": item["flags"], "pred_flags": R.detect_constraints(item["text"]),
                 "text": item["text"]})


def acc(split=None):
    sel = [r for r in rows if split in (None, r["split"])]
    return round(sum(r["pred"] == r["gold"] for r in sel) / len(sel), 3)


out = {"task_accuracy_classifier_only": {"all": acc(), "dev": acc("dev"), "test": acc("test")}}
final_acc = lambda split=None: round(sum(r["final"] == r["gold"] for r in rows if split in (None, r["split"])) /
                                    len([r for r in rows if split in (None, r["split"])]), 3)
out["task_accuracy"] = {"all": final_acc(), "dev": final_acc("dev"), "test": final_acc("test")}
out["per_task"] = {t: round(sum(r["pred"] == t for r in rows if r["gold"] == t) / 12, 2) for t in R.TASKS}
out["confusions"] = Counter(f"{r['gold']}->{r['pred']}" for r in rows if r["pred"] != r["gold"]).most_common(8)
right = [r["conf"] for r in rows if r["pred"] == r["gold"]]
wrong = [r["conf"] for r in rows if r["pred"] != r["gold"]]
out["mean_confidence"] = {"right": round(sum(right) / len(right), 3), "wrong": round(sum(wrong) / max(1, len(wrong)), 3)}
for lo in (0.5, 0.7):
    hi = [r for r in rows if r["conf"] >= lo]
    out[f"accuracy_when_confidence>={lo}"] = {"n": len(hi), "acc": round(sum(r["pred"] == r["gold"] for r in hi) / len(hi), 3)}


def flag_stats(split):
    stats = {}
    sel = [r for r in rows if split in (None, r["split"])]
    for f in ("local_only", "low_latency", "long_text", "negation", "many_labels"):
        def ok(r):  # many_labels must also get the count right
            return f in r["pred_flags"] and (f != "many_labels" or r["pred_flags"][f] == r["flags"].get(f))
        tp = sum(1 for r in sel if f in r["flags"] and ok(r))
        fp = sum(1 for r in sel if f not in r["flags"] and f in r["pred_flags"])
        fn = sum(1 for r in sel if f in r["flags"] and not ok(r))
        stats[f] = {"precision": round(tp / (tp + fp), 2) if tp + fp else None,
                    "recall": round(tp / (tp + fn), 2) if tp + fn else None, "tp": tp, "fp": fp, "fn": fn}
    return stats


out["flags_dev"], out["flags_test"] = flag_stats("dev"), flag_stats("test")
t0 = time.perf_counter()
for r in rows:
    R.recommend(r["text"], {"laya-multilingual-mlx"})
out["ms_per_request"] = round((time.perf_counter() - t0) / len(rows) * 1000, 2)
print(json.dumps(out, ensure_ascii=False, indent=1))
Path(__file__).with_name("results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
Path(__file__).with_name("errors.txt").write_text(
    "\n".join(f"{r['gold']}->{r['pred']} ({r['conf']}) {r['text']}" for r in rows if r["pred"] != r["gold"]), encoding="utf-8")

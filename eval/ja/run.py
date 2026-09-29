"""Japanese evaluation of the local Laya server (Jev-compatible API).

Usage (server must be running): uv run python eval/ja/run.py [--base URL] [--only a,b]
Tasks: latency, choice, label_lang, noul, score, context. Writes eval/ja/results.json.
Only talks to the local server; nothing is sent to Jev.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import httpx

from data import (ALL_DESC, CATEGORIES, CHOICE_ITEMS, DISTRACTORS, FILLER, NOUL_ITEMS,
                  SCORE_ITEMS, SCORE_LEVELS)

MLX, COREML, ANE = "laya-multilingual-mlx", "laya-multilingual-coreml", "laya-multilingual-coreml-ane"


# Target selection via env so every eval script can also run against hosted Jev:
#   EVAL_BASE=https://api.typesafe.ai EVAL_MODEL=jev-latest EVAL_TAG=jev TYPESAFE_API_KEY=... uv run python ...
# EVAL_TAG is appended to result file names so laya results are never overwritten.
import os

TAG = os.environ.get("EVAL_TAG", "")
MODEL_OVERRIDE = os.environ.get("EVAL_MODEL")


def result_name(name: str) -> str:
    return name.replace(".json", f"_{TAG}.json") if TAG else name


class Client:
    def __init__(self, base: str):
        base = os.environ.get("EVAL_BASE", base)
        key = os.environ.get("TYPESAFE_API_KEY") if "127.0.0.1" not in base else None
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.http = httpx.Client(base_url=base, timeout=300, headers=headers)

    def ask(self, model, state, questions):
        model = MODEL_OVERRIDE or model
        t = time.perf_counter()
        for attempt in range(6):
            r = self.http.post("/v1/systemone", json={"state": state, "model": model, "questions": questions})
            if r.status_code != 429:
                break
            time.sleep(2 ** attempt)
        ms = (time.perf_counter() - t) * 1000
        if r.status_code != 200:
            return None, ms, f"{r.status_code} {r.text[:120]}"
        return r.json(), ms, ""


def choice_q(labels):
    return {"q": {"type": "choice", "instructions": "この問い合わせの種類はどれですか？", "criteria": labels}}


def ece(pairs, bins=10):
    total, err = len(pairs), 0.0
    for b in range(bins):
        grp = [(c, k) for c, k in pairs if b / bins < c <= (b + 1) / bins or (b == 0 and c == 0)]
        if grp:
            err += len(grp) / total * abs(sum(k for _, k in grp) / len(grp) - sum(c for c, _ in grp) / len(grp))
    return round(err, 3)


def summarize(pairs, errs):
    n = len(pairs)
    return {"n": n, "errors": errs,
            "acc": round(sum(k for _, k in pairs) / n, 3) if n else None,
            "ece": ece(pairs) if n else None,
            "mean_conf": round(statistics.mean(c for c, _ in pairs), 3) if n else None}


def build_labels(n, gold, rng):
    pool = list(CATEGORIES) + DISTRACTORS
    names = pool[:n] if n >= 5 else [gold] + [x for x in CATEGORIES if x != gold][: n - 1]
    if gold not in names:
        names[-1] = gold
    rng.shuffle(names)
    return {k: ALL_DESC[k] for k in names}


def run_latency(c, out):
    res = {}
    for m in ((MLX,) if MODEL_OVERRIDE else (MLX, COREML, ANE)):
        one = {"q": {"type": "noul", "instructions": "この文は苦情ですか？"}}
        five = {f"q{i}": {"type": "noul", "instructions": f"質問{i}: この文は苦情ですか？"} for i in range(5)}
        for _ in range(5):
            c.ask(m, "商品が壊れていました。", one)
        res[m] = {}
        for label, qs in (("single", one), ("batch5", five)):
            ts = []
            for _ in range(40):
                resp, ms, err = c.ask(m, "商品が壊れていました。", qs)
                if resp is None:
                    res[m][label] = {"error": err}
                    break
                ts.append(ms)
            else:
                ts.sort()
                res[m][label] = {"p50_ms": round(statistics.median(ts), 1), "p95_ms": round(ts[int(len(ts) * .95)], 1)}
    out["latency"] = res


def run_choice(c, out):
    res = {}
    for m in ((MLX,) if MODEL_OVERRIDE else (MLX, COREML, ANE)):
        res[m] = {}
        for n in (2, 5, 10, 20, 50):
            rng = random.Random(n)
            pairs, errs, last = [], 0, ""
            for text, gold in CHOICE_ITEMS:
                resp, _, err = c.ask(m, text, choice_q(build_labels(n, gold, rng)))
                if resp is None:
                    errs, last = errs + 1, err
                    continue
                a = resp["answers"]["q"]
                pairs.append((a["confidence"], a["choice"] == gold))
            res[m][str(n)] = {**summarize(pairs, errs), **({"last_error": last} if errs else {})}
    out["choice_scaling"] = res


def run_label_lang(c, out):
    en = {"配送": ("delivery", "Questions about shipping or delivery status"),
          "返品・返金": ("return_refund", "Questions about returns, exchanges or refunds"),
          "請求・支払い": ("billing", "Questions about charges or payment"),
          "技術トラブル": ("tech_issue", "Technical problems with the app or product"),
          "解約": ("cancel", "Requests to cancel the subscription or account")}
    res = {}
    for mode in ("ja_labels", "en_labels"):
        ok = 0
        for text, gold in CHOICE_ITEMS:
            if mode == "ja_labels":
                labels, want = dict(CATEGORIES), gold
            else:
                labels, want = {v[0]: v[1] for v in en.values()}, en[gold][0]
            resp, _, _ = c.ask(MLX, text, choice_q(labels))
            ok += resp is not None and resp["answers"]["q"]["choice"] == want
        res[mode] = round(ok / len(CHOICE_ITEMS), 3)
    out["label_language"] = res


def run_noul(c, out):
    phr = {"positive": ("この文は他者への攻撃的・侮辱的な内容を含んでいますか？", False),
           "negated": ("この文は攻撃的でも侮辱的でもない、穏当な内容ですか？", True),
           "no_criteria_english": ("Does this text contain aggressive or insulting language toward others?", False)}
    res = {}
    for name, (q, flip) in phr.items():
        pairs, probs = [], {"aggr": [], "benign": []}
        for text, aggressive in NOUL_ITEMS:
            resp, _, _ = c.ask(MLX, text, {"q": {"type": "noul", "instructions": q}})
            p = resp["answers"]["q"]["noul"]
            p_aggr = 1 - p if flip else p
            probs["aggr" if aggressive else "benign"].append(p_aggr)
            pairs.append((max(p_aggr, 1 - p_aggr), (p_aggr >= .5) == aggressive))
        res[name] = {**summarize(pairs, 0),
                     "mean_p_aggr_on_aggr": round(statistics.mean(probs["aggr"]), 3),
                     "mean_p_aggr_on_benign": round(statistics.mean(probs["benign"]), 3)}
    out["noul"] = res


def run_score(c, out):
    xs, ys, argmax = [], [], [0] * 5
    for text, lvl in SCORE_ITEMS:
        q = {"q": {"type": "score", "instructions": "この感想の顧客満足度はどれですか？", "criteria": SCORE_LEVELS}}
        resp, _, _ = c.ask(MLX, text, q)
        a = resp["answers"]["q"]
        xs.append(a["score"]); ys.append(lvl)
        argmax[int(max(a["probabilities"], key=a["probabilities"].get))] += 1
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    r = cov / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** .5
    out["score"] = {"pearson_r": round(r, 3), "mae": round(statistics.mean(abs(x - y) for x, y in zip(xs, ys)), 3),
                    "argmax_level_counts(gold=2 each)": argmax, "scores": [round(x, 2) for x in xs]}


def run_context(c, out):
    probe, *_ = c.ask(MLX, FILLER * 10, choice_q(dict(CATEGORIES)))
    base = c.ask(MLX, "a", choice_q(dict(CATEGORIES)))[0]["usage"]["input_tokens"]
    per = (probe["usage"]["input_tokens"] - base) / 10
    res = {}
    for model, targets in (((MLX, (300, 700, 1000, 2000, 4000, 6000, 8000)),) if MODEL_OVERRIDE else
                           ((MLX, (300, 700, 1000, 2000, 4000, 6000, 8000)), (COREML, (700, 1000, 1300)))):
        res[model] = {}
        for tgt in targets:
            reps = max(0, int((tgt - 200) / per))
            for pos in ("message_last", "message_first"):
                pairs, errs, toks, last = [], 0, [], ""
                items = CHOICE_ITEMS if tgt <= 4000 else CHOICE_ITEMS[::2]
                for text, gold in items:
                    st = FILLER * reps + text if pos == "message_last" else text + FILLER * reps
                    resp, _, err = c.ask(model, st, choice_q(dict(CATEGORIES)))
                    if resp is None:
                        errs, last = errs + 1, err
                        continue
                    a = resp["answers"]["q"]
                    pairs.append((a["confidence"], a["choice"] == gold))
                    toks.append(resp["usage"]["input_tokens"])
                res[model][f"{tgt}/{pos}"] = {**summarize(pairs, errs),
                                              "input_tokens": round(statistics.mean(toks)) if toks else None,
                                              **({"last_error": last} if errs else {})}
    res["tokens_per_filler"] = round(per, 1)
    out["context"] = res


TASKS = {"latency": run_latency, "choice": run_choice, "label_lang": run_label_lang,
         "noul": run_noul, "score": run_score, "context": run_context}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8017")
    ap.add_argument("--only", default=",".join(TASKS))
    a = ap.parse_args()
    path = Path(__file__).with_name(result_name("results.json"))
    out = json.loads(path.read_text()) if path.exists() else {}
    c = Client(a.base)
    for name in a.only.split(","):
        print(f"== {name}", flush=True)
        TASKS[name](c, out)
        print(json.dumps(out[{"label_lang": "label_language", "choice": "choice_scaling"}.get(name, name)],
                         ensure_ascii=False, indent=1), flush=True)
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

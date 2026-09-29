"""Time-series judgments with Laya: raw numbers vs rule-generated Japanese summary.

Synthetic series with known labels. Tasks: trend (上昇/下降/横ばい), spike (noul), next-step direction.
Usage (server running): uv run python eval/ja/run_ts.py
"""

from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

from run import result_name, MLX, Client, summarize

N = 30  # series per class


def make(kind, n, rng):
    base = rng.uniform(50, 150)
    noise = base * 0.03
    xs = []
    for i in range(n):
        if kind == "up":
            v = base + i * base * 0.02
        elif kind == "down":
            v = base - i * base * 0.02
        else:
            v = base
        xs.append(v + rng.gauss(0, noise))
    return xs


def with_spike(xs, rng):
    xs = list(xs)
    i = rng.randint(len(xs) // 2, len(xs) - 1)
    xs[i] *= rng.uniform(1.8, 2.5)
    return xs


def raw(xs):
    return "時系列の値: " + ", ".join(f"{x:.0f}" for x in xs)


def summary(xs):
    n = len(xs)
    first, last = statistics.mean(xs[: n // 4]), statistics.mean(xs[-n // 4:])
    change = (last - first) / first * 100
    peak = max(xs)
    med = statistics.median(xs)
    spike = peak > med * 1.5
    trend = "上昇傾向" if change > 8 else "下降傾向" if change < -8 else "ほぼ横ばい"
    s = f"直近{n}点の推移: 序盤の平均から終盤の平均まで{change:+.0f}%で、{trend}。"
    if spike:
        s += f"最大値は中央値の{peak / med:.1f}倍に達し、突出した値がある。"
    else:
        s += "突出した値はない。"
    return s


def run(c, out):
    rng = random.Random(1)
    res = {}
    for length in (12, 60):
        for rep_name, rep in (("raw", raw), ("summary", summary)):
            # trend (choice)
            pairs, errs = [], 0
            for kind, gold in (("up", "上昇"), ("down", "下降"), ("flat", "横ばい")):
                for _ in range(N):
                    xs = make(kind, length, rng)
                    q = {"q": {"type": "choice", "instructions": "この時系列の全体的な傾向はどれですか？",
                               "criteria": {"上昇": "値が全体として増えている", "下降": "値が全体として減っている",
                                            "横ばい": "値がほぼ一定で大きな変化がない"}}}
                    resp, _, err = c.ask(MLX, rep(xs), q)
                    if resp is None:
                        errs += 1
                        continue
                    a = resp["answers"]["q"]
                    pairs.append((a["confidence"], a["choice"] == gold))
            res[f"trend/{rep_name}/len{length}"] = summarize(pairs, errs)
            # spike (noul)
            pairs = []
            for has in (True, False):
                for _ in range(N):
                    xs = make(rng.choice(["up", "down", "flat"]), length, rng)
                    if has:
                        xs = with_spike(xs, rng)
                    resp, _, _ = c.ask(MLX, rep(xs), {"q": {"type": "noul",
                                       "instructions": "この時系列には突出した異常値(スパイク)が含まれていますか？"}})
                    p = resp["answers"]["q"]["noul"]
                    pairs.append((max(p, 1 - p), (p >= .5) == has))
            res[f"spike/{rep_name}/len{length}"] = summarize(pairs, 0)
    # next-step direction: latent trend + noise; baseline = sign of last delta
    for rep_name, rep in (("raw", raw), ("summary", summary)):
        pairs, base_ok = [], 0
        for _ in range(3 * N):
            slope = rng.choice([-1, 0, 1]) * 0.01
            base = rng.uniform(50, 150)
            xs = [base * (1 + slope * i) + rng.gauss(0, base * 0.03) for i in range(13)]
            hist, nxt = xs[:12], xs[12]
            gold_up = nxt > hist[-1]
            base_ok += (hist[-1] > hist[-2]) == gold_up
            resp, _, _ = c.ask(MLX, rep(hist), {"q": {"type": "noul", "instructions": "次の値は直近の値より大きくなりますか？"}})
            p = resp["answers"]["q"]["noul"]
            pairs.append((max(p, 1 - p), (p >= .5) == gold_up))
        res[f"next_up/{rep_name}"] = {**summarize(pairs, 0), "baseline_last_delta_acc": round(base_ok / (3 * N), 3)}
    out["timeseries"] = res


if __name__ == "__main__":
    path = Path(__file__).with_name(result_name("results_ts.json"))
    out = {}
    run(Client("http://127.0.0.1:8017"), out)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out["timeseries"].items():
        print(k, v)

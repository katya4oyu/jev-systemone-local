"""Compile results_*_<tag>.json of all evaluated models into markdown tables (printed to stdout).

Tags: laya (MLX base, no suffix), ft (fine-tuned laya), jev, jeff08, jeff2b, kev08, kev4b.
"""
import json
from pathlib import Path

here = Path(__file__).parent
MODELS = [("laya", ""), ("laya-ft", "ft"), ("jev", "jev"), ("jeff-0.8B", "jeff08"), ("jeff-2B", "jeff2b"), ("kev-0.8B", "kev08"), ("kev-4B", "kev4b"), ("clef-flash-8bit", "clefflash8")]


def load(base, tag):
    name = base.replace(".json", f"_{tag}.json") if tag else base
    p = here / name
    return json.loads(p.read_text()) if p.exists() else None


def first(d):  # single-model dicts keyed by model name
    return next(iter(d.values())) if d else None


def g(d, *ks, default="-"):
    for k in ks:
        if d is None:
            return default
        d = d.get(k) if isinstance(d, dict) else None
    return default if d is None else d


def fmt(x):
    return f"{x:.2f}" if isinstance(x, float) else str(x)


def table(title, rows, header):
    print(f"\n### {title}\n")
    print("| " + " | ".join(["指標"] + header) + " |")
    print("|" + "---|" * (len(header) + 1))
    for name, vals in rows:
        print("| " + " | ".join([name] + [fmt(v) for v in vals]) + " |")


R = {n: {} for n, _ in MODELS}
for n, tag in MODELS:
    R[n]["main"] = load("results.json", tag)
    R[n]["dialog"] = load("results_dialog.json", tag)
    R[n]["ts"] = load("results_ts.json", tag)
    R[n]["wiki"] = load("results_wiki.json", tag)
    R[n]["wsyn"] = load("results_wiki_synth.json", tag)
    R[n]["wvar"] = load("results_wiki_variants.json", tag)
    R[n]["reg"] = load("results_regression.json", tag)
names = [n for n, _ in MODELS]


def choice_acc(n, k):
    m = R[n]["main"] or R[n]["reg"]
    c = first(m.get("choice_scaling", {})) if m else None
    v = c.get(str(k)) if c else None
    if not v:
        return "-"
    return v["acc"] if v["acc"] is not None else f"err({v['errors']})"


rows = [(f"5分類 {k}ラベル 正答率", [choice_acc(n, k) for n in names]) for k in (2, 5, 10, 20, 50)]
table("日本語の問い合わせ5分類(24件)", rows, names)


def noul_acc(n, k):
    m = R[n]["main"] or R[n]["reg"]
    return g(m, "noul", k, "acc") if m else "-"


table("毒性 noul 正答率(20件)", [(k, [noul_acc(n, k) for n in names]) for k in ("positive", "negated", "no_criteria_english")], names)
table("score(感想5段階)", [("Pearson r", [g(R[n]["main"], "score", "pearson_r") for n in names])], names)


def ctx(n, key):
    m = R[n]["main"]
    c = first(g(m, "context", default={}) and {k: v for k, v in m["context"].items() if k != "tokens_per_filler"}) if m else None
    v = c.get(key) if c else None
    if not v:
        return "-"
    return v["acc"] if v["acc"] is not None else f"err({v['errors']})"


table("文脈長(本文が末尾)正答率", [(k, [ctx(n, k) for n in names]) for k in ("300/message_last", "2000/message_last", "4000/message_last", "8000/message_last")], names)


def lat(n, k):
    m = R[n]["main"]
    l = first(g(m, "latency", default={})) if m else None
    return g(l, k, "p50_ms") if l else "-"


table("遅延 p50(ms、クライアント計測。jev はネットワーク込)", [("単発", [lat(n, "single") for n in names]), ("5問バッチ", [lat(n, "batch5") for n in names])], names)


def d(n, t, f, key="auc"):
    return g(R[n]["dialog"], t, f, key)


rows = []
for t, label in (("end_of_utterance", "発話区切り"), ("memory_worthy", "記憶すべきか"), ("topic_drift", "話題の逸脱")):
    rows.append((f"{label} AUC(2択)", [d(n, t, "choice2") for n in names]))
    rows.append((f"{label} AUC(noul)", [d(n, t, "noul") for n in names]))
rows.append(("応答/相槌/無反応 正答率", [g(R[n]["dialog"], "react_choice", "acc") for n in names]))
table("会話ターン処理", rows, names)

ts = lambda n, k: g(R[n]["ts"], "timeseries", k, "acc")
table("時系列(正答率)", [(k, [ts(n, k) for n in names]) for k in ("trend/raw/len12", "trend/raw/len60", "spike/raw/len12", "trend/summary/len60", "spike/summary/len60", "next_up/raw")], names)
table("実在wiki(AUC)", [("リンク妥当性(長い入力)", [g(R[n]["wiki"], "link", "auc") for n in names]),
                       ("同一センス", [g(R[n]["wiki"], "sense", "auc") for n in names]),
                       ("リンク: 短い入力 noul", [g(R[n]["wvar"], "noul_short", "auc") for n in names]),
                       ("ページ種別 memos 正答率", [g(R[n]["wiki"], "type_memos", "acc") for n in names])], names)
table("合成wiki(AUC、2択)", [(k, [g(R[n]["wsyn"], k, "choice2", "auc") for n in names]) for k in ("link", "split", "duplicate")], names)

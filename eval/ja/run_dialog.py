"""Turn-handling judgments for voice/chat agents: end-of-utterance, react type, memory-worthiness, topic drift.

Each binary task is asked in two forms (noul vs 2-way choice) since form matters (see run_wiki_variants.py).
Usage (server running): uv run python eval/ja/run_dialog.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from data_dialog import DRIFT_ITEMS, EOU_ITEMS, MEMORY_ITEMS, REACT_ITEMS
from run import result_name, MLX, Client, summarize


def auc(pos, neg):
    return round(sum((a > b) + .5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg)), 3)


def binary(c, rows, form):
    """rows: [(state, gold_bool)]; form(state) -> (question, extractor)."""
    ps = []
    for state, gold in rows:
        q, ext = form()
        resp, _, _ = c.ask(MLX, state, {"q": q})
        ps.append((ext(resp["answers"]["q"]), gold))
    pos = [p for p, g in ps if g]
    neg = [p for p, g in ps if not g]
    accs = [(sum((p >= t) == g for p, g in ps) / len(ps), t) for t in [i / 20 for i in range(1, 20)]]
    best = max(accs)
    # false-positive rate on negatives at the threshold giving >=90% recall on positives
    rec90 = [t for t in [i / 40 for i in range(1, 40)] if sum(p >= t for p in pos) / len(pos) >= .9]
    fpr = round(sum(p >= max(rec90) for p in neg) / len(neg), 3) if rec90 else None
    return {"n": len(ps), "acc@0.5": round(sum((p >= .5) == g for p, g in ps) / len(ps), 3), "auc": auc(pos, neg),
            "best_thr_acc": [round(best[0], 3), best[1]], "fpr_at_recall90": fpr,
            "mean_p_pos": round(statistics.mean(pos), 3), "mean_p_neg": round(statistics.mean(neg), 3)}


def two_forms(question, yes_desc, no_desc, yes="はい", no="いいえ"):
    def noul_form():
        return {"type": "noul", "instructions": question}, lambda a: a["noul"]

    def choice_form():
        return ({"type": "choice", "instructions": question, "criteria": {yes: yes_desc, no: no_desc}},
                lambda a: a["probabilities"][yes])
    return {"noul": noul_form, "choice2": choice_form}


def main():
    c = Client("http://127.0.0.1:8017")
    out = {}

    eou = [(t, ok) for t, ok in EOU_ITEMS]
    forms = two_forms("この発話者は、ここまでで話し終えていますか？", "発話は完結しており、相手が応答してよい", "まだ話の途中で、続きがありそう",
                      "完結", "継続中")
    out["end_of_utterance"] = {k: binary(c, eou, f) for k, f in forms.items()}

    # partial fragments: cut complete utterances at 50-70% to measure false "finished" on truncated text
    cut = [(t[: max(4, int(len(t) * .6))], False) for t, ok in EOU_ITEMS if ok and len(t) >= 12]
    full = [(t, True) for t, ok in EOU_ITEMS if ok]
    out["eou_truncation"] = {k: binary(c, cut + full, f) for k, f in forms.items()}

    labels = {"応答": "AIが今すぐ答えるべき(質問・依頼・呼びかけ)", "相槌": "ユーザーが話している最中で、AIは相槌だけ打つべき",
              "無反応": "独り言やAI宛でない発話で、AIは反応しない方がよい"}
    pairs, errs, conf = [], 0, {}
    for text, gold in REACT_ITEMS:
        resp, _, _ = c.ask(MLX, text, {"q": {"type": "choice", "instructions": "AIアバターはこの発話にどう反応すべきですか？", "criteria": labels}})
        a = resp["answers"]["q"]
        pairs.append((a["confidence"], a["choice"] == gold))
        conf[f"{gold}->{a['choice']}"] = conf.get(f"{gold}->{a['choice']}", 0) + 1
    out["react_choice"] = {**summarize(pairs, errs), "confusion": dict(sorted(conf.items()))}

    mem = [(t, w) for t, w in MEMORY_ITEMS]
    forms = two_forms("この発言には、今後も覚えておくべき個人的な事実・好み・予定が含まれていますか？", "長く覚えておくべき情報がある",
                      "その場限りの雑談や一時的な内容", "覚える", "覚えない")
    out["memory_worthy"] = {k: binary(c, mem, f) for k, f in forms.items()}

    drift = [("\n".join(h) + f"\nユーザー(新しい発言): {new}", on) for h, new, on in DRIFT_ITEMS]
    forms = two_forms("最後の「新しい発言」は、直前までの会話と同じ話題の続きですか？", "同じ話題の続き", "別の話題に移った、または話題が分岐した",
                      "続き", "別話題")
    out["topic_drift"] = {k: binary(c, drift, f) for k, f in forms.items()}

    Path(__file__).with_name(result_name("results_dialog.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, v in out.items():
        print(k, json.dumps(v, ensure_ascii=False))


if __name__ == "__main__":
    main()

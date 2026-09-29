"""Does the pairwise failure depend on the question form? Compare formulations on the same link/sense cases.

Usage (server running): uv run python eval/ja/run_wiki_variants.py /path/to/wiki/notes
Aggregate numbers only (results_wiki_variants.json).
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

from run import result_name, MLX, Client
from run_wiki import LINK, load, paragraphs


def auc(pos, neg):
    return round(sum((a > b) + .5 * (a == b) for a in pos for b in neg) / max(1, len(pos) * len(neg)), 3)


def score(c, cases, form):
    """form: (state_fn, question dict builder, extractor of P(match))."""
    ps = []
    for case, gold in cases:
        state, q, ext = form(case)
        resp, _, _ = c.ask(MLX, state, {"q": q})
        ps.append((ext(resp["answers"]["q"]), gold))
    pos = [p for p, g in ps if g]
    neg = [p for p, g in ps if not g]
    acc = sum((p >= .5) == g for p, g in ps) / len(ps)
    best = max(sum((p >= t) == g for p, g in ps) / len(ps) for t in [i / 20 for i in range(1, 20)])
    return {"auc": auc(pos, neg), "acc@0.5": round(acc, 3), "best_thr_acc": round(best, 3),
            "mean_p_pos": round(sum(pos) / len(pos), 3), "mean_p_neg": round(sum(neg) / len(neg), 3)}


def main():
    root = Path(sys.argv[1])
    know = load(root, "knowledge")
    rng = random.Random(0)
    names = list(know)
    cases = []
    for name, pg in know.items():
        for para in paragraphs(pg["body"]):
            for m in LINK.finditer(para):
                tgt = m.group(2)[2:]
                if tgt in know and tgt != name:
                    cases.append((name, m.group(1), LINK.sub(r"\1", para), tgt))
    rng.shuffle(cases)
    cases = cases[:80]
    pairs = []
    for name, anchor, snip, tgt in cases:
        linked = {m.group(2)[2:] for m in LINK.finditer(know[name]["body"])} | {name, tgt}
        hard = [x for x in names if x not in linked and know[x]["tags"] & know[tgt]["tags"]] or \
               [x for x in names if x not in linked]
        neg = rng.choice(hard)
        pairs.append(((snip, anchor, know[tgt]), True))
        pairs.append(((snip, anchor, know[neg]), False))

    def f_noul_full(case):
        s, a, t = case
        return f"{s}\n\n候補ページ: {t['title']} — {t['summary']}", {"type": "noul", "instructions": "この文章は候補ページの概念に言及していますか？"}, lambda a_: a_["noul"]

    def f_noul_short(case):
        s, a, t = case
        return f"言及: {a}\n候補ページ: {t['title']}", {"type": "noul", "instructions": "言及は候補ページと同じ概念を指していますか？"}, lambda a_: a_["noul"]

    def f_choice_pair(case):
        s, a, t = case
        crit = {"同じ概念": "言及が候補ページと同じ概念を指している", "別の概念": "言及は候補ページとは別の概念を指している"}
        return f"言及: {a}\n候補ページ: {t['title']} — {t['summary']}", {"type": "choice", "instructions": "言及と候補ページの関係はどれですか？", "criteria": crit}, lambda a_: a_["probabilities"]["同じ概念"]

    def f_choice_pair_ctx(case):
        s, a, t = case
        crit = {"同じ概念": "言及が候補ページと同じ概念を指している", "別の概念": "言及は候補ページとは別の概念を指している"}
        return f"文脈: {s[:300]}\n言及: {a}\n候補ページ: {t['title']} — {t['summary']}", {"type": "choice", "instructions": "「言及」と候補ページの関係はどれですか？", "criteria": crit}, lambda a_: a_["probabilities"]["同じ概念"]

    def f_score(case):
        s, a, t = case
        return f"言及: {a}\n候補ページ: {t['title']} — {t['summary']}", {"type": "score", "instructions": "言及と候補ページの関連の強さはどれですか？", "criteria": ["無関係", "弱い関連", "同じ概念"]}, lambda a_: a_["score"] / 2

    c = Client("http://127.0.0.1:8017")
    out = {}
    for name, f in (("noul_full", f_noul_full), ("noul_short", f_noul_short), ("choice_pair", f_choice_pair),
                    ("choice_pair_ctx", f_choice_pair_ctx), ("score_relatedness", f_score)):
        out[name] = score(c, pairs, f)
        print(name, out[name], flush=True)
    Path(__file__).with_name(result_name("results_wiki_variants.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

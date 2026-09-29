"""Build RLCD training items (tokenized sequences + gold distributions) for Japanese turn handling.

Tasks: end-of-utterance (EOU), react type, memory-worthiness, topic drift. Each labeled row is turned into
several *views* (noul form and 2-way-choice form, several wordings) so the model does not overfit one phrasing.
Hard labels become one-hot gold distributions (no Jev outputs are used as targets).

Usage: python build_items.py <base_checkpoint_dir> <out_dir> [--seed 0]
Writes out_dir/train_items.pt and out_dir/dev_items.pt (10% of rows held out by row, never trained on).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "data"))
from laya.agent import _fix_tokenizer_config  # noqa: E402
from laya.common import QTYPES, build_sequence, render_options  # noqa: E402

from train_drift import TRAIN_DRIFT  # noqa: E402
from train_eou import TRAIN_EOU  # noqa: E402
from train_memory import TRAIN_MEMORY  # noqa: E402
from train_react import TRAIN_REACT  # noqa: E402

# view = (type, instructions, criteria, key of the "positive" option). The first noul / first choice view of each
# task is the wording used by eval/ja/run_dialog.py; the rest are paraphrases. Eval also probes wordings that
# are NOT here (run_dialog_wordings.py) to measure generalisation.
VIEWS = {
    "eou": [
        ("noul", "この発話者は、ここまでで話し終えていますか？", None, None),
        ("noul", "発話は完結していますか？", None, None),
        ("noul", "話し手はもう言い終わっていますか？", None, None),
        ("choice", "この発話者は、ここまでで話し終えていますか？",
         {"完結": "発話は完結しており、相手が応答してよい", "継続中": "まだ話の途中で、続きがありそう"}, "完結"),
        ("choice", "話し手の状態はどちらですか？",
         {"話し終わり": "言い終えていて、次は聞き手の番", "話の途中": "言いかけで、まだ続きを話しそう"}, "話し終わり"),
    ],
    "memory": [
        ("noul", "この発言には、今後も覚えておくべき個人的な事実・好み・予定が含まれていますか？", None, None),
        ("noul", "この発言は、長期的に記憶しておく価値がありますか？", None, None),
        ("noul", "話し手についての、今後も役立つ持続的な情報が含まれていますか？", None, None),
        ("choice", "この発言には、今後も覚えておくべき個人的な事実・好み・予定が含まれていますか？",
         {"覚える": "長く覚えておくべき情報がある", "覚えない": "その場限りの雑談や一時的な内容"}, "覚える"),
        ("choice", "この発言の扱いはどちらですか？",
         {"記憶する": "個人的な事実・好み・予定・人間関係など、後でも使える情報", "忘れる": "雑談・反応・一回限りの依頼など"}, "記憶する"),
    ],
    "drift": [
        ("noul", "最後の「新しい発言」は、直前までの会話と同じ話題の続きですか？", None, None),
        ("noul", "新しい発言は、これまでの話題を引き継いでいますか？", None, None),
        ("noul", "会話の話題は変わらずに続いていますか？", None, None),
        ("choice", "最後の「新しい発言」は、直前までの会話と同じ話題の続きですか？",
         {"続き": "同じ話題の続き", "別話題": "別の話題に移った、または話題が分岐した"}, "続き"),
        ("choice", "最後の発言と、それまでの会話の関係は？",
         {"同じ話題": "前の話題を続けている", "話題転換": "前の話題から離れて別の話題を始めた"}, "同じ話題"),
    ],
}
REACT_LABELS = [
    {"応答": "AIが今すぐ答えるべき(質問・依頼・呼びかけ)", "相槌": "ユーザーが話している最中で、AIは相槌だけ打つべき",
     "無反応": "独り言やAI宛でない発話で、AIは反応しない方がよい"},
    {"応答": "AI宛の質問・依頼なので、今答える", "相槌": "ユーザーが話の途中なので、うなずくだけにする",
     "無反応": "AIに向けた発言ではないので、何もしない"},
]


def drift_state(history, new):
    return "\n".join(history) + f"\nユーザー(新しい発言): {new}"


def rows():
    """yield (task, state, gold) with gold either bool or class name."""
    for t, ok in TRAIN_EOU:
        yield "eou", t, ok
    for t, w in TRAIN_MEMORY:
        yield "memory", t, w
    for h, n, on in TRAIN_DRIFT:
        yield "drift", drift_state(h, n), on
    for t, g in TRAIN_REACT:
        yield "react", t, g


def make_views(task, state, gold, rng, n_views):
    out = []
    if task == "react":
        for _ in range(n_views):
            crit = rng.choice(REACT_LABELS)
            ins = rng.choice(["AIアバターはこの発話にどう反応すべきですか？", "この発話に対してAIはどう振る舞うべきですか？"])
            keys = list(crit)
            out.append(({"type": "choice", "instructions": ins, "criteria": crit}, [1.0 if k == gold else 0.0 for k in keys]))
        return out
    noul = [v for v in VIEWS[task] if v[0] == "noul"]
    choice = [v for v in VIEWS[task] if v[0] == "choice"]
    picks = [rng.choice(noul), rng.choice(choice)][:n_views]
    for typ, ins, crit, pos in picks:
        if typ == "noul":
            out.append(({"type": "noul", "instructions": ins}, [0.0, 1.0] if gold else [1.0, 0.0]))
        else:
            keys = list(crit)
            out.append(({"type": "choice", "instructions": ins, "criteria": crit},
                        [1.0 if (k == pos) == bool(gold) else 0.0 for k in keys]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    _fix_tokenizer_config(a.base)
    tok = AutoTokenizer.from_pretrained(os.path.join(a.base, "tokenizer"))
    cfg = json.load(open(os.path.join(a.base, "rl_agent_config.json")))
    all_rows = list(rows())
    rng.shuffle(all_rows)
    n_dev = int(len(all_rows) * a.dev_frac)
    splits = {"dev": all_rows[:n_dev], "train": all_rows[n_dev:]}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for name, rs in splits.items():
        items, skipped = [], 0
        for task, state, gold in rs:
            for q, target in make_views(task, state, gold, rng, 2):
                t = q["type"]
                crit = q.get("criteria") or {}
                seq, markers = build_sequence(tok, state, {"t": t, "ins": q["instructions"], "crit": crit},
                                              cfg["max_len"], cfg["head_max_len"])
                if len(markers) != len(render_options({"t": t, "crit": crit})):
                    skipped += 1
                    continue
                items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t], "target": target,
                              "label": target.index(max(target)), "task": task})
        torch.save(items, Path(a.out) / f"{name}_items.pt")
        by = {}
        for it in items:
            by[it["task"]] = by.get(it["task"], 0) + 1
        print(name, len(items), "items", by, "skipped", skipped)


if __name__ == "__main__":
    main()

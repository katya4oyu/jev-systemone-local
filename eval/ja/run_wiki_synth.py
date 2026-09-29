"""Hand-written (synthetic) wiki pairs: link / split / duplicate, noul vs 2-way choice. Server must be running."""
from __future__ import annotations
import json
from pathlib import Path
from data_wiki import DUP_ITEMS, SPLIT_ITEMS, WIKI_PAIRS
from run import result_name, Client
from run_dialog import binary, two_forms

def main():
    c = Client("http://127.0.0.1:8017")
    out = {}
    link = [(f"言及: {s}\n候補ページ: {t} — {sm}", g) for s, t, sm, g in WIKI_PAIRS]
    f = two_forms("言及は候補ページと同じ概念を指していますか？", "言及が候補ページの概念を指している", "言及は候補ページとは別の概念", "同じ概念", "別の概念")
    out["link"] = {k: binary(c, link, v) for k, v in f.items()}
    split = [(f"段落1: {a}\n段落2: {b}", g) for a, b, g in SPLIT_ITEMS]
    f = two_forms("段落2は、段落1と同じ一つの概念の説明の続きですか？", "同じ概念の続き", "別の概念が始まっている", "続き", "別概念")
    out["split"] = {k: binary(c, split, v) for k, v in f.items()}
    dup = [(f"ページA: {a}\nページB: {b}", g) for a, b, g in DUP_ITEMS]
    f = two_forms("ページAとページBは同じ概念を扱う重複ページですか？", "同じ概念(重複)", "関連はあるが別の概念", "重複", "別概念")
    out["duplicate"] = {k: binary(c, dup, v) for k, v in f.items()}
    Path(__file__).with_name(result_name("results_wiki_synth.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for t, d in out.items():
        for k, v in d.items():
            print(t, k, "auc", v["auc"], "acc@.5", v["acc@0.5"], "best", v["best_thr_acc"], "pos/neg", v["mean_p_pos"], v["mean_p_neg"])
if __name__ == "__main__":
    main()

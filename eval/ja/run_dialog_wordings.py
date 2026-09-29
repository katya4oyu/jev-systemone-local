"""Sanity check: is the near-chance dialog result due to one unlucky wording? Try several wordings per task."""
from __future__ import annotations
import json
from pathlib import Path
from data_dialog import EOU_ITEMS, MEMORY_ITEMS, DRIFT_ITEMS
from run import result_name, Client
from run_dialog import binary

def choice(question, yes, no, yes_d, no_d):
    return lambda: ({"type": "choice", "instructions": question, "criteria": {yes: yes_d, no: no_d}}, lambda a: a["probabilities"][yes])

def noul(question):
    return lambda: ({"type": "noul", "instructions": question}, lambda a: a["noul"])

def main():
    c = Client("http://127.0.0.1:8017")
    out = {}
    eou = list(EOU_ITEMS)
    W = {
     "noul_短い": noul("話し終えた？"),
     "noul_文が完結": noul("この文は文として完結していますか？"),
     "noul_続く": noul("この発話にはまだ続きがありますか？"),
     "choice_言い終わり/言いかけ": choice("この発話の状態は？", "言い終わり", "言いかけ", "文が終わっている", "文の途中で終わっている"),
     "choice_完全な文/途切れた文": choice("この文字列はどちらですか？", "完全な文", "途切れた文", "意味が最後まで通る文", "途中で切れて意味が完結しない文"),
    }
    out["eou"] = {k: binary(c, eou, f) for k, f in W.items()}
    mem = list(MEMORY_ITEMS)
    W = {
     "noul_個人情報": noul("これは話し手についての個人的な情報ですか？"),
     "noul_覚えて": noul("この内容は、後で思い出せるよう記憶に保存すべきですか？"),
     "choice_個人的/雑談": choice("この発言の種類は？", "個人的な事実", "雑談", "話し手の家族・住まい・予定・好みなど個人的な事実", "天気や相槌などその場限りの雑談"),
    }
    out["memory"] = {k: binary(c, mem, f) for k, f in W.items()}
    drift = [("\n".join(h) + f"\nユーザー(新しい発言): {new}", on) for h, new, on in DRIFT_ITEMS]
    W = {
     "noul_話題が変わった(反転)": lambda: ({"type": "noul", "instructions": "最後の発言で話題が変わりましたか？"}, lambda a: 1 - a["noul"]),
     "noul_同じ話題": noul("最後の発言は、ここまでと同じ話題についてですか？"),
     "choice_同/別": choice("最後の発言は？", "同じ話題", "別の話題", "前の話題を続けている", "前の話題から離れて別の話題を始めた"),
    }
    out["drift"] = {k: binary(c, drift, f) for k, f in W.items()}
    Path(__file__).with_name(result_name("results_dialog_wordings.json")).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for t, d in out.items():
        for k, v in d.items():
            print(t, k, "auc", v["auc"], "acc@.5", v["acc@0.5"], "best", v["best_thr_acc"], "pos/neg", v["mean_p_pos"], v["mean_p_neg"])

if __name__ == "__main__":
    main()

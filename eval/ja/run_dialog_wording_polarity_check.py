"""Focused corrected-polarity check for the EOU continuation question.

Runs the existing 60 EOU_ITEMS against the explicitly selected model and writes
aggregate-only output to a separate file; it never overwrites historical wording results.
"""
from __future__ import annotations

import json
from pathlib import Path

from data_dialog import EOU_ITEMS
from run import Client, result_name
from run_dialog import binary
from run_dialog_wordings import noul_for_finished

QUESTION = "この発話にはまだ続きがありますか？"


def main() -> None:
    c = Client("http://127.0.0.1:8017")
    metrics = binary(c, list(EOU_ITEMS), noul_for_finished(QUESTION))
    result = {
        "task": "end_of_utterance",
        "question": QUESTION,
        "scored_label": "finished",
        "n": len(EOU_ITEMS),
        **metrics,
    }
    path = Path(__file__).with_name(result_name("results_dialog_wordings_eou_continue_fixed.json"))
    path.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()

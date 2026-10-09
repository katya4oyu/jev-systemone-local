"""Verify five-suite result counts against source datasets and task-server counters."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_ROOT = ROOT.parent.parent if ROOT.parent.name == ".worktrees" else ROOT
EVAL = ROOT / "eval" / "ja"
sys.path.insert(0, str(EVAL))
from data import CHOICE_ITEMS, CATEGORIES, NOUL_ITEMS, SCORE_ITEMS  # noqa: E402
from data_dialog import DRIFT_ITEMS, EOU_ITEMS, MEMORY_ITEMS, REACT_ITEMS  # noqa: E402
from data_wiki import DUP_ITEMS, SPLIT_ITEMS, WIKI_PAIRS  # noqa: E402

CONTEXT_TARGETS = (300, 700, 1000, 2000, 4000, 6000, 8000)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def dictionaries(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, list):
        for child in value:
            yield from dictionaries(child)


def sum_metric(value: Any, key: str) -> int:
    return sum(int(obj[key]) for obj in dictionaries(value) if isinstance(obj.get(key), (int, float)))


def count_n(value: Any) -> int:
    return sum_metric(value, "n")


def count_errors(value: Any) -> int:
    return sum_metric(value, "errors")


def case_successes(name: str, result: dict[str, Any]) -> int:
    reported = count_n(result)
    if name == "main":
        # run.py reports the two label-language rates and score vectors without n fields.
        reported += 2 * len(CHOICE_ITEMS)
        reported += len(result["score"]["scores"])
    return reported


def expected_cases() -> dict[str, int]:
    context = sum(
        (len(CHOICE_ITEMS) if target <= 4000 else len(CHOICE_ITEMS[::2])) * 2
        for target in CONTEXT_TARGETS
    )
    trunc = [
        (text[: max(4, int(len(text) * 0.6))], False)
        for text, complete in EOU_ITEMS if complete and len(text) >= 12
    ]
    full = [(text, True) for text, complete in EOU_ITEMS if complete]
    return {
        "main": 5 * len(CHOICE_ITEMS) + 2 * len(CHOICE_ITEMS)
                + 3 * len(NOUL_ITEMS) + len(SCORE_ITEMS) + context,
        "timeseries": 2 * 2 * 3 * 30 + 2 * 2 * 2 * 30 + 2 * 3 * 30,
        "dialog": 2 * len(EOU_ITEMS) + 2 * len(trunc + full) + len(REACT_ITEMS)
                  + 2 * len(MEMORY_ITEMS) + 2 * len(DRIFT_ITEMS),
        "wordings": 5 * len(EOU_ITEMS) + 3 * len(MEMORY_ITEMS) + 3 * len(DRIFT_ITEMS),
        "wiki_synth": 2 * (len(WIKI_PAIRS) + len(SPLIT_ITEMS) + len(DUP_ITEMS)),
    }


def expected_requests() -> tuple[dict[str, int], dict[str, int]]:
    main_requests = (
        5 + 40 * 2  # five single warmups; 40 single and 40 batch5 latency observations
        + 5 * len(CHOICE_ITEMS)
        + 2 * len(CHOICE_ITEMS)
        + 3 * len(NOUL_ITEMS)
        + len(SCORE_ITEMS)
        + 2  # filler calibration and short prompt calibration
        + sum(
            (len(CHOICE_ITEMS) if target <= 4000 else len(CHOICE_ITEMS[::2])) * 2
            for target in CONTEXT_TARGETS
        )
    )
    main_questions = (
        5 + 40 + 40 * 5  # latency requests have one or five named questions
        + 5 * len(CHOICE_ITEMS)
        + 2 * len(CHOICE_ITEMS)
        + 3 * len(NOUL_ITEMS)
        + len(SCORE_ITEMS)
        + 2
        + sum(
            (len(CHOICE_ITEMS) if target <= 4000 else len(CHOICE_ITEMS[::2])) * 2
            for target in CONTEXT_TARGETS
        )
    )
    requests = {
        "main": main_requests,
        "timeseries": 2 * 2 * 3 * 30 + 2 * 2 * 2 * 30 + 2 * 3 * 30,
        "dialog": expected_cases()["dialog"],
        "wordings": expected_cases()["wordings"],
        "wiki_synth": expected_cases()["wiki_synth"],
    }
    questions = {name: count for name, count in requests.items()}
    questions["main"] = main_questions
    return requests, questions


def first_model(result: dict[str, Any], section: str) -> dict[str, Any]:
    candidates = result[section]
    for value in candidates.values():
        if isinstance(value, dict):
            return value
    raise ValueError(f"missing model result in {section}")


def metric_vector(files: dict[str, Path]) -> dict[str, Any]:
    main = load_json(files["main"])
    dialog = load_json(files["dialog"])
    wordings = load_json(files["wordings"])
    wiki = load_json(files["wiki"])
    latency = first_model(main, "latency")
    choices = first_model(main, "choice_scaling")
    context = first_model(main, "context")
    dialog_eou = dialog["end_of_utterance"]["choice2"]
    eou_wordings = [entry["auc"] for entry in wordings["eou"].values()]
    context_lengths = [
        int(value["input_tokens"])
        for key, value in context.items()
        if "/" in key and value.get("n", 0) > 0 and value.get("input_tokens") is not None
    ]
    return {
        "latency_single_p50_ms": latency["single"]["p50_ms"],
        "latency_batch5_p50_ms": latency["batch5"]["p50_ms"],
        "latency_batch5_p95_ms": latency["batch5"]["p95_ms"],
        "choice_50_accuracy": choices["50"]["acc"],
        "japanese_label_accuracy": main["label_language"]["ja_labels"],
        "english_label_accuracy": main["label_language"]["en_labels"],
        "noul_negated_accuracy": main["noul"]["negated"]["acc"],
        "score_mae": main["score"]["mae"],
        "eou_choice_auc": dialog_eou["auc"],
        "eou_choice_acc_at_0_5": dialog_eou["acc@0.5"],
        "eou_wording_auc_min": min(eou_wordings),
        "eou_wording_auc_max": max(eou_wordings),
        "react_choice_accuracy": dialog["react_choice"]["acc"],
        "memory_choice_auc": dialog["memory_worthy"]["choice2"]["auc"],
        "topic_drift_choice_auc": dialog["topic_drift"]["choice2"]["auc"],
        "wiki_link_choice_auc": wiki["link"]["choice2"]["auc"],
        "max_successful_context_usage_tokens": max(context_lengths),
    }


def historical_sources() -> dict[str, dict[str, Path]]:
    strands = CANONICAL_ROOT / ".worktrees" / "strands-decider-v21" / "eval" / "ja"
    tag = "strandsv21mlx_20261009"
    return {
        "Kev-4B": {
            "main": EVAL / "results_kev4b.json",
            "dialog": EVAL / "results_dialog_kev4b.json",
            "wordings": EVAL / "results_dialog_wordings_kev4b.json",
            "wiki": EVAL / "results_wiki_synth_kev4b.json",
        },
        "Clef-Flash-8B": {
            "main": EVAL / "results_clefflash8.json",
            "dialog": EVAL / "results_dialog_clefflash8.json",
            "wordings": EVAL / "results_dialog_wordings_clefflash8.json",
            "wiki": EVAL / "results_wiki_synth_clefflash8.json",
        },
        "Strands-Decider-v2.1": {
            "main": strands / f"results_{tag}.json",
            "dialog": strands / f"results_dialog_{tag}.json",
            "wordings": strands / f"results_dialog_wordings_{tag}.json",
            "wiki": strands / f"results_wiki_synth_{tag}.json",
        },
    }


def build(tag: str, before_path: Path, after_path: Path) -> dict[str, Any]:
    names = {
        "main": f"results_{tag}.json",
        "timeseries": f"results_ts_{tag}.json",
        "dialog": f"results_dialog_{tag}.json",
        "wordings": f"results_dialog_wordings_{tag}.json",
        "wiki_synth": f"results_wiki_synth_{tag}.json",
    }
    results = {name: load_json(EVAL / filename) for name, filename in names.items()}
    expected = expected_cases()
    observed_cases = {name: case_successes(name, value) for name, value in results.items()}
    result_errors = {name: count_errors(value) for name, value in results.items()}
    actual_completed = {name: observed_cases[name] + result_errors[name] for name in names}
    if actual_completed != expected:
        raise AssertionError(f"suite case count mismatch; completed={actual_completed}, expected={expected}")

    request_counts, question_counts = expected_requests()
    before, after = load_json(before_path), load_json(after_path)
    deltas = {
        key: int(after[key]) - int(before[key])
        for key in ("systemone_requests", "successful_requests", "rejected_requests", "runtime_errors", "questions")
    }
    type_deltas = {
        key: int(after["question_types"][key]) - int(before["question_types"][key])
        for key in ("choice", "noul", "score")
    }
    expected_requests_total = sum(request_counts.values())
    expected_questions_total = sum(question_counts.values())
    if deltas["systemone_requests"] != expected_requests_total:
        raise AssertionError(f"HTTP call count mismatch: {deltas['systemone_requests']} != {expected_requests_total}")
    if deltas["successful_requests"] != expected_requests_total:
        raise AssertionError(f"successful call count mismatch: {deltas['successful_requests']} != {expected_requests_total}")
    if deltas["questions"] != expected_questions_total:
        raise AssertionError(f"question count mismatch: {deltas['questions']} != {expected_questions_total}")
    if deltas["rejected_requests"] or deltas["runtime_errors"]:
        raise AssertionError(f"server recorded errors: {deltas}")

    current_files = {
        "main": EVAL / names["main"],
        "dialog": EVAL / names["dialog"],
        "wordings": EVAL / names["wordings"],
        "wiki": EVAL / names["wiki_synth"],
    }
    comparison = {"LiquidAI/d1-3B": metric_vector(current_files)}
    history_paths = historical_sources()
    for model, files in history_paths.items():
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"historical result missing for {model}: {missing}")
        comparison[model] = metric_vector(files)

    total_cases = sum(observed_cases.values())
    return {
        "tag": tag,
        "model": after["model"],
        "suite_results": {
            name: {
                "file": str((EVAL / filename).relative_to(ROOT)),
                "successful_cases": observed_cases[name],
                "reported_case_errors": result_errors[name],
                "completed_cases": actual_completed[name],
                "expected_cases": expected[name],
                "script_requests_expected": request_counts[name],
            }
            for name, filename in names.items()
        },
        "totals": {
            "suite_files": len(names),
            "completed_cases": total_cases + sum(result_errors.values()),
            "successful_cases": total_cases,
            "reported_case_errors": sum(result_errors.values()),
            "expected_http_requests": expected_requests_total,
            "observed_http_requests": deltas["systemone_requests"],
            "observed_successful_requests": deltas["successful_requests"],
            "rejected_requests": deltas["rejected_requests"],
            "runtime_errors": deltas["runtime_errors"],
            "expected_questions": expected_questions_total,
            "observed_questions": deltas["questions"],
            "observed_question_types": type_deltas,
            "max_prepared_input_tokens_during_suites": int(after["max_prepared_input_tokens"]),
            "max_response_usage_input_tokens_during_suites": int(after["max_response_usage_input_tokens"]),
            "input_context_limit": int(after["model"]["max_input_tokens"]),
        },
        "comparison": comparison,
        "comparison_sources": {
            "Kev-4B": "origin/main@8e714df8dbf7acffe098e686295a10df74b1d9ef:eval/ja/results*_kev4b.json",
            "Clef-Flash-8B": "origin/main@8e714df8dbf7acffe098e686295a10df74b1d9ef:eval/ja/results*_clefflash8.json",
            "Strands-Decider-v2.1": "eval/strands-decider-v21@8775969b63b3a29ad91b2d77bd311e2e1a341f53:eval/ja/results*strandsv21mlx_20261009.json",
        },
        "stats_baseline": str(before_path.relative_to(ROOT)),
        "stats_after": str(after_path.relative_to(ROOT)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.before = args.before.resolve()
    args.after = args.after.resolve()
    args.output = args.output.resolve()
    result = build(args.tag, args.before, args.after)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Run the whole Japanese eval suite against any Jev-compatible server.
# Usage: eval/ja/run_all.sh <base_url> <model> <tag> [wiki_notes_dir]
# Results: eval/ja/results_*_<tag>.json. Only pass wiki_notes_dir for a LOCAL server (it sends wiki text).
set -euo pipefail
cd "$(dirname "$0")"
export EVAL_BASE="$1" EVAL_MODEL="$2" EVAL_TAG="$3"
PY=${PYTHON:-"uv run python"}
$PY run.py --only latency,choice,label_lang,noul,score,context
$PY run_ts.py
$PY run_dialog.py
$PY run_dialog_wordings.py
$PY run_wiki_synth.py
if [ -n "${4:-}" ]; then
  $PY run_wiki.py "$4"
  $PY run_wiki_variants.py "$4"
fi

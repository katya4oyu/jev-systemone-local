# Strands Decider v21 MLX: ローカル日本語評価

**状態: モデル取得、MLX serve、API/SDK smoke、既存日本語 suite、workbench pytest、cleanup まで完了。** 2026-10-09。評価 tag: `strandsv21mlx_20261009`。この評価は Apple M4 Pro 一台での単回・ローカル実測で、一般性能や品質の保証ではない。

## 対象と固定した provenance

- systemone-workbench 基準: `origin/main` / `8e714df8dbf7acffe098e686295a10df74b1d9ef`。専用 worktree `/Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/strands-decider-v21`、branch `eval/strands-decider-v21`。canonical main と既存 worktree は変更していない。
- 公式実装: [`strands-labs/strands-decider`](https://github.com/strands-labs/strands-decider)、commit `63d24ae286e50105fba0bd1db15b0e243aea4650`。
- Official checkpoint: `StrandsAgents/strands-decider-2B-hobson-v21`、revision `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`。base: `Qwen/Qwen3.5-2B-Base`、revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`（checkpoint provenance の revision）。Model card / official project license は Apache-2.0。
- `hf cache verify` は checkpoint の27ファイルと base の12ファイルすべて成功。LoRA adapter SHA-256 `59be987f5eb664a7f74f11ef69383a0de526fe30c684932102984eb9c11f4028`、head SHA-256 `0fc78684d7504d334082d6ac8e6cb7825b5eb0e230601181cd273757f9f6ebef`。checkpoint snapshot 91,846,359 bytes、base snapshot 4,571,206,192 bytes。
- 重み・公式 source checkout・venv は workbench repository 外の durable path `/Users/yuya/.local/share/systemone-evals/strands-decider-v21/` に保持。Hermes の Python 環境は変更していない。host は macOS arm64 / Apple M4 Pro、RAM 51,539,607,552 bytes、開始時の空き容量 144 GiB。
- Python 3.12.12 task venv。導入版: strands-decider `0.0.1.dev16+g63d24ae28`、torch 2.7.1、transformers 5.17.0、peft 0.21.0、mlx 0.32.3、mlx-lm 0.32.0、huggingface-hub 1.33.0、`uv` 0.12.0。公式 source の `[mlx]` extra と inference docs の Mac pins を利用。

## Serve・実推論 smoke

`127.0.0.1:8024` のみで PID 16622 が推論中に稼働した。command は task venv の公式 `strands-decider serve`、`--device mlx --strict-window --model-name strands-decider-2B-hobson-v21`。環境は task-owned `HF_HOME` に限定し、`HF_HUB_OFFLINE=1` と `TRANSFORMERS_OFFLINE=1` を設定。初回起動時に既定 cache と task-owned cache のパスを取り違えて `FileNotFoundError` となったが、task-owned pinned snapshot に直して起動・推論に成功した（初回は listener を開いていない）。

- `GET /health`: HTTP 200。`status=ok`、model `strands-decider-2B-hobson-v21`、base `Qwen/Qwen3.5-2B-Base`、device `mlx`、`max_length=4096`、`vision=false`。checkpoint path は pinned snapshot。
- `/v1/systemone` に明示 model 名で日本語の noul/choice/score を1問ずつ送信。HTTP 200、応答 model も同名。実測: noul `0.9096`、choice `返金`（配送 0.0602、返金 0.8132、技術 0.1266）、score `1.2182`（3段階）。usage 239 input / 3 output tokens、server latency 1184.46 ms。
- Python `typesafe-sdk 0.7.1` から同じ named model / 3 question types の実推論に成功し、型付き noul/choice/score response を確認。SDK のローカル接続用 key は `local-evaluation` という dummy 値（server は認証なし）、secret ではない。
- `GET /v1/models` は HTTP 404 (`Not Found`)。この endpoint は未提供として記録し、model-list 互換とは主張しない。
- 評価終了後 PID 16622 へ SIGTERM。Uvicorn は shutdown complete を記録。PID 消滅、port 8024 の listener なし、後続 `/health` は接続失敗（curl exit 7）を確認。

## workbench pytest

```text
env -u PYTHONPATH -u PYTHONHOME uv --no-config sync --extra test
# exit 0
env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q
# 79 passed, 1 Starlette/anyio BlockingPortal deprecation warning; exit 0
```

## 既存日本語 suite の結果

`eval/ja/run_all.sh` を第4引数（私的 wiki directory）なしで実行。標準の5 script がすべて終了し、exit 0。以下の case 数と error 数は5 JSONを Python で parse/集計した値である。case 数はスコア対象または context request の項目数で、context 拒否も含む。latency の warmup/反復と context の校正用2 request は除外。model key が既存 harness 上 `laya-multilingual-mlx` に固定された JSON もあるが、`EVAL_MODEL=strands-decider-2B-hobson-v21` override を使用し、別途実応答の model 名も確認した。

| suite | cases | errors | 結果 |
|---|---:|---:|---|
| `run.py`（latency / choice / label_lang / noul / score / context） | 526 | 48 | exit 0。48件すべて strict-window の想定内 HTTP 422 |
| `run_ts.py` | 780 | 0 | exit 0 |
| `run_dialog.py` | 413 | 0 | exit 0 |
| `run_dialog_wordings.py` | 540 | 0 | exit 0 |
| `run_wiki_synth.py` | 260 | 0 | exit 0 |
| **合計** | **2,519** | **48** | **`run_all.sh` exit 0** |

### 主な指標

- Latency は HTTP client 側計測（5回 warmup 後）。単発 p50/p95 **57.0/57.8 ms**、5問 batch **186.1/189.5 ms**。短い状態でのこの Mac・この run のみ。
- Choice 24件ずつ。候補数2/5/10/20/50の accuracy は **0.958/0.958/0.958/0.958/0.917**、API error 0。日本語/英語ラベルは両方 **0.958**。
- Noul positive/negated/English は各20件 accuracy **1.000**、API error 0。ECE は順に **0.143/0.232/0.128** で、accuracy と confidence calibration は別に読む。
- Score は10件、Pearson r **0.965**、MAE **0.49**。小標本の傾向値で、品質保証ではない。
- Context 4096 token window。約300〜4000 target の240件はすべて応答。約4000条件（平均 3,886 input tokens）は本文末尾 accuracy **0.958**、本文先頭 **1.000**。約6000 target は12件×本文順2通りとも 422（prompt 5,905 tokens）、約8000 target も同じく422（7,925 tokens）。計48件の拒否は `--strict-window` の挙動であり、切り詰められた回答ではない。
- 時系列780件: summary形式のtrend accuracyは12/60点とも **1.000**、raw形式は **0.967/0.989**。spikeはraw **0.933/0.833**、summary **1.000/0.833**。next-step direction はraw **0.489**、summary **0.433** と弱い。
- 会話413件: EOU choice AUC **0.909**（n=60）だが切断を含む EOU は choice AUC **0.631**（n=44）。応答/相槌/無反応 choice accuracy **0.356**（n=45）。topic drift は AUC **0.993/0.995**（noul/choice、各40件）、memory-worthy は AUC **0.959/0.925** だが threshold 0.5 の accuracy は **0.55/0.525**。
- Wording 540件: 文の完結性 noul accuracy/AUC **0.900/0.971**、短い「話し終えた？」は **0.567/0.811**。`noul_続く` は共通 gold の「finished」へ確率を反転する既存 `noul_for_finished()` を通る。topic drift の反転 wording は AUC **0.525** とほぼ chance。一律な言い換え耐性とは言えない。
- 合成 wiki 260件: link / split / duplicate の AUC は noul **0.941/0.975/0.964**、choice **0.921/1.000/0.996**。私的 wiki は送信していない。

この suite は小規模な自作・合成セットで、単回実行。モデル比較や統計的有意差、私的 wiki の品質、4k超 context、vision、並列負荷は検証していない。特に 4k 上限と会話/next-step の弱い項目を考慮し、本測定だけから既定モデル採用を勧めない。

## 結果 artifact

この branch に集計元JSONを保持。どれもテスト指標のみで、raw input/state、私的 wiki、credential は含まない。

- `eval/ja/results_strandsv21mlx_20261009.json`
- `eval/ja/results_ts_strandsv21mlx_20261009.json`
- `eval/ja/results_dialog_strandsv21mlx_20261009.json`
- `eval/ja/results_dialog_wordings_strandsv21mlx_20261009.json`
- `eval/ja/results_wiki_synth_strandsv21mlx_20261009.json`

## 再現

source checkout、task venv、cache は `/Users/yuya/.local/share/systemone-evals/strands-decider-v21/` に保持している。Weights は再ダウンロード済みなので `hf download` は省略可能。初回構築時の pins:

```bash
ROOT=/Users/yuya/.local/share/systemone-evals/strands-decider-v21
cd "$ROOT/strands-decider"
env -u PYTHONPATH -u PYTHONHOME uv venv --python 3.12 "$ROOT/venv"
env -u PYTHONPATH -u PYTHONHOME uv pip install --python "$ROOT/venv/bin/python" \
  -e "$ROOT/strands-decider[mlx]" torch==2.7.1 transformers==5.17.0 peft==0.21.0
HF_HOME="$ROOT/hf-home" "$ROOT/venv/bin/hf" download StrandsAgents/strands-decider-2B-hobson-v21 \
  --revision 2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5
HF_HOME="$ROOT/hf-home" "$ROOT/venv/bin/hf" download Qwen/Qwen3.5-2B-Base \
  --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c
```

serve（task 完了時は停止済み）:

```bash
ROOT=/Users/yuya/.local/share/systemone-evals/strands-decider-v21
CKPT="$ROOT/hf-home/hub/models--StrandsAgents--strands-decider-2B-hobson-v21/snapshots/2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5"
env -u PYTHONPATH -u PYTHONHOME HF_HOME="$ROOT/hf-home" HF_HUB_DISABLE_TELEMETRY=1 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$ROOT/venv/bin/strands-decider" serve "$CKPT" \
  --host 127.0.0.1 --port 8024 --device mlx --strict-window \
  --model-name strands-decider-2B-hobson-v21
```

workbench checkout で test と全標準 suite を再実行:

```bash
cd /Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/strands-decider-v21
env -u PYTHONPATH -u PYTHONHOME uv --no-config sync --extra test
env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q
env -u PYTHONPATH -u PYTHONHOME \
  EVAL_BASE=http://127.0.0.1:8024 EVAL_MODEL=strands-decider-2B-hobson-v21 \
  EVAL_TAG=strandsv21mlx_20261009 \
  PYTHON="$PWD/.venv/bin/python" \
  ./eval/ja/run_all.sh http://127.0.0.1:8024 strands-decider-2B-hobson-v21 strandsv21mlx_20261009
```

Suite を再実行する場合、tag が既存 JSON を上書きするため、新しい `EVAL_TAG` を指定する。

# Clef-Flash MLX 4bit：日本語評価・実測レポート

**進行中（2026-10-04 03:14 JST）**。固定revisionの4bit重み取得・metadata確認は完了し、worktreeのrepo testsは65 passed。評価suiteはまだ開始していない。5回のtask-owned server起動はすべてEADDRINUSEで終了（最初の4回はmodel load後、5回目はport予約bind時）。同一Python processからのIPv4/IPv6 loopback/wildcard bindも8021で失敗する一方、lsof/netstatにはsocket ownerが見えず、Clef/eval processもない。8022/8023/8031はdirect bind可能と確認。重複評価runは起動していない。Mainの指定portを独断で変えず、8021の予約境界を解消するか8022へ切り替える判断を待つ。

## モデルprovenance

- Hugging Face repo：`mlx-community/clef-flash-4bit`
- 固定revision：`140bf7e037f5fa95a96535feca112f46c15927cf`
- Base：`Cloudflare/clef-flash`（公式revision `17f0b0ad64efb65d273590632833508766b2aae6`）
- Snapshot：`/Users/yuya/.cache/huggingface/hub/models--mlx-community--clef-flash-4bit/snapshots/140bf7e037f5fa95a96535feca112f46c15927cf`
- `config.json`：`bits=4`, `group_size=64`, `mode=affine`。backboneは9B。model shard 2個とjoint headの合計は実ファイルで `6,193,757,576 bytes`（約6.19 GB）。公式モデルカードの丸め表記は6.2 GB。
- custom `clef_mlx.py` は8bit固定revisionの同名loaderとSHA-256 `852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5` が一致。loaderはtext generationではなくjoint schema headを実行する。量子化差はconfig上の4bit/8bit設定。

## 実行環境

- Python 3.12.12、arm64 Apple Silicon、RAM `51,539,607,552 bytes`、MLX default device `Device(gpu, 0)`、Metal available。
- 既存隔離環境 `/Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv` を再利用。mlx 0.32.3、mlx-lm 0.32.0、mlx-vlm 0.7.4、transformers 5.18.0、huggingface-hub 1.33.0、httpx 0.28.1、typesafe-sdk 0.7.2。環境は書き換えていない。
- 8021番は通常server起動4回とモデルload前の予約bind 1回がEADDRINUSE。4回目は100.38秒観測してsocketなし。直接bind probeでもIPv4/IPv6のloopback/wildcardすべて失敗したが、lsof/netstatには該当socketがなく、実行中の所有processを特定できなかった。8022/8023/8031はbind可能。別processは停止していない。

## smoke・評価suite・比較の状態

- REST choice/noul/score実推論、`typesafe_sdk`型付き実推論、model名のrequest/response一致、`--no-truncate`の16,384上限超過413確認：**未実行**。8021のbind blocker解決後に実行する。
- 既存suite（`run_all.sh`：core latency/choice/label-language/noul/score/context、timeseries、dialog、dialog wordings、wiki synth、private wiki、wiki variants）：**未開始**。
- 採点修正確認の60件 `run_dialog_wording_polarity_check.py`：**未開始**。
- 4bit / 8bit / historical Kev-4B比較：**未生成**。Kev-4Bは既存記録のみで再測定しない。
- `compare.py`には `clef-flash-4bit` / `clefflash4`列を追加済み。4bit result JSONとsuite完了後に比較表を生成・確認する。
- `uv --no-config sync --extra test` exit 0。独立worktreeのPython 3.13.15環境で `env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q` は **65 passed, 0 failed**、Starlette/anyioのDeprecationWarning 1件。最終treeで再確認する。

## phase log・task-owned resource

- append-only phase log：`/Users/yuya/.hermes/cache/scratch/clef_flash_4bit_eval.log`
- 新規評価runnerはまだ起動していないため、重複runはない。
- phase collectorは初回のlog file置換で停止（inode変更）。以後は専用append helperで同じinodeへ追記。collector PID 42043は再起動後readiness確認済みで、`PHASE_EXIT`通知のみ。今後logのtruncate/atomic replaceはしない。
- server / watcherの所有PID、起動・終了phase、port消失は完了時に記録する。

## 比較と解釈上の注意

- `run_dialog_wordings.py`の「まだ続きがありますか？」raw値はgold極性との不一致で無効。8bit作業で追加済みの `noul_for_finished()` を用いて、60件のfocused再評価結果を比較する。
- 会話項目は質問の意味が違う条件（例：個人情報判定と記憶すべきか）を単なる言い換え耐性とまとめない。
- 既存Kev-4B値はhistorical baselineであり、今回のhardware・実行環境では再測定しない。
- 自作の少数評価・私的wikiのFrontmatterを機械的goldに用いるため、結果はモデルの実用保証・統計的有意差ではない。wiki本文はlocalhost推論のみで扱い、本文やprivate snippetは保存せず集計JSONだけを残す。

## 再現手順

```bash
SNAPSHOT=/Users/yuya/.cache/huggingface/hub/models--mlx-community--clef-flash-4bit/snapshots/140bf7e037f5fa95a96535feca112f46c15927cf
env -u PYTHONPATH -u PYTHONHOME \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I \
  "$SNAPSHOT/clef_mlx.py" serve --model "$SNAPSHOT" \
  --name clef-flash-4bit --host 127.0.0.1 --port 8021 --no-truncate
```

別terminalで同じ既存suiteと修正極性のfocused確認を実行する（完了後に実際のexit code/metricsでこの欄を更新する）。

```bash
cd /Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/clef-flash-4bit
env -u PYTHONPATH -u PYTHONHOME \
  EVAL_BASE=http://127.0.0.1:8021 EVAL_MODEL=clef-flash-4bit EVAL_TAG=clefflash4 \
  PYTHON="/Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python /Users/yuya/.hermes/cache/scratch/clef_flash_4bit_phase_wrapper.py" \
  ./eval/ja/run_all.sh http://127.0.0.1:8021 clef-flash-4bit clefflash4 \
  /Users/yuya/src/github.com/katya4oyu/me/notes

env -u PYTHONPATH -u PYTHONHOME \
  EVAL_BASE=http://127.0.0.1:8021 EVAL_MODEL=clef-flash-4bit EVAL_TAG=clefflash4 \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I \
  eval/ja/run_dialog_wording_polarity_check.py
```

Repository checksは独立したworktree環境で実行する。

```bash
uv --no-config sync --extra test
env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q
git diff --check
```

## 最終状態（未完）

- worktree/branch：`/Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/clef-flash-4bit` / `eval/clef-flash-4bit`、base `241c55b07ffe2fc07137154779556f2e993104b4`。
- 初期checkpoint commit：`4725f79806970d277e39e62b184f136cea52fc70`（reportとcomparison列）。このcommitのworktreeはclean。suite完了後の最終結果commitは未作成。
- suite result JSON、smoke、比較結果、final HEAD、server/watcher cleanup：**未完了**。repo testsは現時点で65 passed（1 deprecation warning）、最終treeでも再確認する。8021のport境界に関するMain判断後に評価を続ける。

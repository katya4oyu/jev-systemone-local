# Strands Decider v21 MLX: ローカル日本語評価

**状態: 推論環境・API/SDK smoke・pytest 完了。既存日本語 suite は実行中。** 2026-10-09 時点。評価 tag は `strandsv21mlx_20261009`。この記録は suite 完了後に結果・失敗数・cleanup 状態を追記する。

## 対象と固定した provenance

- systemone-workbench の基準: `origin/main` / `8e714df8dbf7acffe098e686295a10df74b1d9ef`。専用 worktree `/Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/strands-decider-v21`、branch `eval/strands-decider-v21`。既存 main / worktree は変更していない。
- 公式実装: [`strands-labs/strands-decider`](https://github.com/strands-labs/strands-decider)、commit `63d24ae286e50105fba0bd1db15b0e243aea4650`。
- Official checkpoint: `StrandsAgents/strands-decider-2B-hobson-v21`、revision `2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`。base: `Qwen/Qwen3.5-2B-Base`、revision `b1485b2fa6dfa1287294f269f5fb618e03d52d7c`（checkpoint provenance の revision）。Model card / official project license は Apache-2.0。
- `hf cache verify` は checkpoint の27ファイルと base の12ファイルで成功。LoRA adapter SHA-256 `59be987f5eb664a7f74f11ef69383a0de526fe30c684932102984eb9c11f4028`、head SHA-256 `0fc78684d7504d334082d6ac8e6cb7825b5eb0e230601181cd273757f9f6ebef`。checkpoint snapshot 91,846,359 bytes、base snapshot 4,571,206,192 bytes。
- 重み・source・venv は workbench repository の外に保持: `/Users/yuya/.local/share/systemone-evals/strands-decider-v21/`。Hermes の Python 環境は変更していない。macOS arm64 / Apple M4 Pro、RAM 51,539,607,552 bytes、評価開始前空き容量 144 GiB。
- Python 3.12.12 task venv。実際に解決・導入した主要版: torch 2.7.1、transformers 5.17.0、peft 0.21.0、mlx 0.32.3、mlx-lm 0.32.0、huggingface-hub 1.33.0。`uv` 0.12.0。Source の `[mlx]` extra と公式 inference docs の Mac pins を使用。

## Serve・実推論 smoke

`127.0.0.1:8024` のみで task-owned PID 16622 が稼働中。server cwd は `/Users/yuya/.local/share/systemone-evals/strands-decider-v21`。CLI は task venv の `strands-decider serve <pinned-checkpoint-snapshot> --host 127.0.0.1 --port 8024 --device mlx --strict-window --model-name strands-decider-2B-hobson-v21`。`HF_HOME` は task-owned cache、`HF_HUB_OFFLINE=1` と `TRANSFORMERS_OFFLINE=1` を設定し、checkpoint/base revision を固定したまま実行。

- `GET /health`: HTTP 200。`status=ok`、model `strands-decider-2B-hobson-v21`、base `Qwen/Qwen3.5-2B-Base`、device `mlx`、`max_length=4096`、`vision=false`、checkpoint は上記 pinned snapshot。
- `/v1/systemone` に明示 model 名で日本語の noul/choice/score 3問を送信。HTTP 200、応答 model も同名。実測: noul `0.9096`、choice `返金`（確率: 配送 0.0602、返金 0.8132、技術 0.1266）、score `1.2182`（3段階）、usage 239 input / 3 output tokens、server latency 1184.46 ms。
- 公式 Python `typesafe-sdk 0.7.1` から同じ named model / 3 question types の実推論に成功し、型付き noul/choice/score response を検証。SDK のローカル接続用 API key は `local-evaluation` という dummy 値（server は認証なし）、secret ではない。
- `GET /v1/models` は HTTP 404 (`Not Found`)。この server では未提供の endpoint として別記し、model-list 互換とは主張しない。

## pytest

```text
env -u PYTHONPATH -u PYTHONHOME uv --no-config sync --extra test
# exit 0
env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q
# 79 passed, 1 Starlette/anyio BlockingPortal deprecation warning; exit 0
```

## 日本語 suite

実行コマンドと結果 JSON は次節・評価完了時に記録する。`eval/ja/run_all.sh` の第4引数（私的 wiki ディレクトリ）は渡さず、公開/合成の標準 suite のみを実行する。全 suite 完了前に総合評価とは扱わない。

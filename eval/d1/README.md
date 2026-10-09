# LiquidAI d1-3B local evaluation adapter

`server.py` is a task-only HTTP adapter for the official `LiquidAI/d1-3B` Python API. It is **not** a LiquidAI server or production integration. It passes requests to the checkpoint's native `model.system_one(...)` method; it does not use text generation, an MLX conversion, or a substitute model.

## Pinned local setup

- Official snapshot: `LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432`
- Local snapshot: `/Users/yuya/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432`
- Runtime: CPython 3.12.12, PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.14.1, FastAPI 0.141.1, uvicorn 0.53.0, HTTPX 0.28.1, TypeSafe SDK 0.7.1
- All Python packages are isolated from Hermes and from the repository's product environment.
- `requirements.txt` pins the direct packages; install into a durable task-owned environment outside the checkout:

```sh
uv --no-config venv --python 3.12 /Users/yuya/.local/share/systemone-d1-3b-eval
uv --no-config pip install \
  --python /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python \
  -r eval/d1/requirements.txt
```

The pre-existing official HF snapshot is reused. No model weights are copied into this worktree.

Run the native smoke separately before starting the HTTP server. This verifies the owner's Python API directly and writes the actual output to `eval/d1/native_smoke.json`:

```sh
env -u PYTHONPATH -u PYTHONHOME \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python \
  eval/d1/native_smoke.py --output eval/d1/native_smoke.json
```

## Start, probe, evaluate

From the repository root, with no ambient Python overrides:

```sh
env -u PYTHONPATH -u PYTHONHOME \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python \
  eval/d1/server.py --host 127.0.0.1 --port 8025
```

The adapter refuses to start unless the pinned snapshot loads with all parameters on MPS in bfloat16. It requires an explicit `model` equal to `liquidai-d1-3b-pytorch-mps`; this is a local adapter alias, not a vendor model name. `/healthz`, `/health`, and `/v1/models` identify the checkpoint and runtime. `/v1/systemone` checks the full, tokenizer-prepared input against the checkpoint's 32,768-token limit before inference and rejects overflow without truncating. `/_task/stats` reports aggregate counts and token maxima only; it never records request text.

The adapter and both real-service smoke scripts refuse non-loopback URLs. The official SDK requires a non-empty API key even for a local override; `sdk_smoke.py` supplies the literal `local-evaluation-no-auth` placeholder, which is ignored by this adapter and is not a credential.

Run the real HTTP contract smoke and TypeSafe Python SDK smoke with all three question types:

```sh
env -u PYTHONPATH -u PYTHONHOME \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python eval/d1/http_smoke.py --base-url http://127.0.0.1:8025
env -u PYTHONPATH -u PYTHONHOME \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python eval/d1/sdk_smoke.py --base-url http://127.0.0.1:8025
```

After these smokes and before running the suites, capture the counter baseline. Capture `stats_after.json` in the same way after the suite so the aggregator verifies the suite-only count deltas:

```sh
curl --fail --silent http://127.0.0.1:8025/_task/stats > eval/d1/stats_before.json
```

Run the same five Japanese suite scripts and save fresh results (replace the tag only when intentionally starting a new run):

```sh
PYTHON=/Users/yuya/.local/share/systemone-d1-3b-eval/bin/python \
  ./eval/ja/run_all.sh \
  http://127.0.0.1:8025 liquidai-d1-3b-pytorch-mps liquid_d1_mps_20261010
```

The HTTP adapter and all evaluation requests must stay loopback-only. After the five scripts finish, save the final counters, run the source-count aggregator, then stop only the server process started for this evaluation and verify that its listener is gone:

```sh
curl --fail --silent http://127.0.0.1:8025/_task/stats > eval/d1/stats_after.json
/Users/yuya/.local/share/systemone-d1-3b-eval/bin/python eval/d1/aggregate.py \
  --tag liquid_d1_mps_20261010 --before eval/d1/stats_before.json \
  --after eval/d1/stats_after.json --output eval/d1/suite_summary.json
```

## Tests

```sh
env -u PYTHONPATH -u PYTHONHOME \
  /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python -m pytest -q eval/d1/test_server.py
```

The contract tests use a fake runtime and verify explicit model selection, the typed response boundary, and exact-limit/one-token-over behavior. They do not replace the separate real-model smoke and suite run.

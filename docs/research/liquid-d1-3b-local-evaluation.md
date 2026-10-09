# LiquidAI/d1-3B local evaluation on Apple M4 Pro

**Run:** 2026-10-10 JST · **worktree:** `eval/liquid-d1-3b` · **base:** `origin/main` at `8e714df8dbf7acffe098e686295a10df74b1d9ef`

**Scope:** official LiquidAI checkpoint, native PyTorch/MPS decision API, the five default Japanese `systemone-workbench` suites, historical comparison. This is an evaluation record, not a production integration or a model recommendation.

## Result

The official `LiquidAI/d1-3B` checkpoint loaded and ran its real `system_one` and `system_one_batch` methods on MPS in bfloat16. All five existing suites completed: **2,519/2,519 scored cases**, **2,606/2,606 HTTP calls**, and **2,766/2,766 named questions**; zero suite errors, rejections, or runtime exceptions. The longest prepared sequence during the suites was **7,910 tokens**, below the checkpoint's 32,768-token context limit; the context suite itself reported up to 7,905 input tokens.

D1's warm HTTP p50 was 49.5 ms for one question and 127.0 ms for five questions over one state. It tied Clef-Flash-8B at 0.833 on 50-way choice accuracy, matched the other historical models on the explicitly negated toxicity question (1.000), and scored between Kev/Clef and Strands on several conversational tasks. Its score MAE (0.281) was worse than Kev-4B (0.194) and Clef-Flash-8B (0.162), but better than Strands Decider v2.1 (0.490). These are small hand-written task-suite observations, not a general model ranking.

## Checkpoint and host evidence

| Item | Verified value |
|---|---|
| Official model | `LiquidAI/d1-3B`, 3.12B parameters, based on `LiquidAI/LFM2.5-VL-3B` |
| Immutable revision | `051bcc464b01b9f92942b364d9586b0ef5912432` (HF API `lastModified`: 2026-10-07) |
| Model config | `lfm2_vl`; text context `max_position_embeddings=32768`; text + vision |
| Weights | `model.safetensors`, 6,247,065,504 bytes |
| Weight SHA-256 | `50e03317847caf6df9a9aee27ed40f20554a86a21e60d1d47ba41a422b546c0c`; matched the pinned HF tree's `lfs_sha256` and `lfs_size` |
| Snapshot reused | `/Users/yuya/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432` (no duplicate copy into the repo) |
| Host | Apple M4 Pro, macOS 27.0, arm64, 51,539,607,552 bytes unified memory (48 GiB); 150.3 GB container space free at a later probe |
| Runtime | CPython 3.12.12, PyTorch 2.14.1, torchvision 0.29.1, Transformers 5.14.1; FastAPI 0.141.1, uvicorn 0.53.0, HTTPX 0.28.1, TypeSafe SDK 0.7.1, Pillow 12.3.0 |
| Device/dtype assertion | `torch.backends.mps.is_available() == True`; every model parameter was on MPS and `torch.bfloat16`; CPU fallback is rejected |

Pinned source links: [model card](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/README.md), [config](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/config.json), [license](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE), [revision tree](https://huggingface.co/LiquidAI/d1-3B/tree/051bcc464b01b9f92942b364d9586b0ef5912432).

The inspected upstream code defines `AutoModel.from_pretrained(..., trust_remote_code=True)` as `D1Model`, whose specialized `system_one` and `system_one_batch` use one-pass typed decision readouts. The evaluation loaded the local immutable snapshot with `local_files_only=True`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`, using `dtype=torch.bfloat16` and SDPA. No generation substitute, MLX conversion, or modified upstream model code was used.

## Native, HTTP, and SDK smoke evidence

- Native `model.system_one(state, named_questions)` returned noul, choice, and score answers over one Japanese state, with 151 input tokens and **0 output tokens**. It chose `返品・返金` (0.9924) and returned the official expected score (0.3884 on the zero-based 0–4 scale). Model-load time was 11.037 s; this first native request took 895.9 ms and is not a warmed latency measurement.
- Native `model.system_one_batch(requests)` processed six examples in 560.1 ms. The Japanese-positive/English-negative, hostile/benign, and explicitly negated benign/hostile checks all matched their expected 0.5-side on the observed noul values (6/6); each answer reported zero output tokens.
- Loopback HTTP `GET /healthz` and `/v1/models` identified the pinned revision, `mps`, `torch.bfloat16`, and 32,768 context. A real `POST /v1/systemone` returned all three typed answer shapes and `model=liquidai-d1-3b-pytorch-mps`; a mismatched requested model was rejected with HTTP 422.
- Official Python SDK 0.7.1 smoke used explicit model selection and obtained noul/choice/score answers with `input_tokens=165`, `output_tokens=0`. SDK response `model` matched the requested alias, and choice/score option probabilities each summed to one. The SDK requires a non-empty API key even for a local URL; the loopback-only smoke script passes the literal `local-evaluation-no-auth` placeholder, which this adapter ignores. No credential was used.

The response probabilities and confidence values retain the official meanings: noul is P(yes), choice confidence is the top option probability, and score is the zero-based expected level. They are **not** treated here as calibrated correctness guarantees.

## Five Japanese suites

The unchanged `eval/ja/run_all.sh` ran its default five scripts without the optional private-wiki argument. Results use a fresh tag; no existing result file was overwritten.

| Suite/script | Scored cases | Errors | Selected observations |
|---|---:|---:|---|
| Main (`run.py`: latency, choice, label language, noul, score, context) | 526 | 0 | 50-option choice accuracy 0.833; Japanese/English label accuracy 1.000/0.958; noul positive/negated/English accuracy 0.850/1.000/0.900; score Pearson r 0.977, MAE 0.281 |
| Time series (`run_ts.py`) | 780 | 0 | Trend accuracy: raw len12 0.989, summary len12 1.000, raw len60 1.000, summary len60 1.000. Spike accuracy: raw len12 0.950, summary len12 1.000, raw len60 0.800, summary len60 0.833. Next-up: raw 0.367, summary 0.433; last-delta baselines 0.389/0.367. |
| Dialog (`run_dialog.py`) | 413 | 0 | End-of-utterance choice AUC/accuracy@0.5 0.896/0.783 (n=60); truncated-vs-full choice AUC 0.783 (n=44); react-choice accuracy 0.689 (n=45); memory choice AUC 0.978; topic-drift choice AUC 1.000. |
| Dialog alternate wordings (`run_dialog_wordings.py`) | 540 | 0 | EOU AUC range 0.672–0.919 across five phrasings; memory AUC 0.846–0.951; drift AUC 0.989–1.000. These prompts differ in wording and, for inverted questions, score polarity is mapped back to the shared label by the existing script; do not average them as interchangeable labels. |
| Synthetic wiki judgments (`run_wiki_synth.py`) | 260 | 0 | Link/split/duplicate choice AUC 0.982/0.998/1.000. |
| **Total** | **2,519** | **0** | All five JSONs and the adapter counters matched source-derived expected counts. |

Latency semantics: the existing client times the full `POST /v1/systemone` round trip with `perf_counter` (HTTP + server + model). It performs five single-question warmup calls, then 40 timed single calls and 40 timed five-question calls. Results were **single p50/p95 49.5/50.6 ms**, **batch-5 p50/p95 127.0/129.7 ms**. These are warm resident decision-call times, not model load time, first-token latency, or generation throughput; d1 emits no output tokens.

For context verification, the task adapter calls the official engine's tokenizer-based `tokens(state, parsed_questions)` on the complete prepared prompt before inference. It rejects overflow instead of truncating. The context suite's longest successful usage was 7,905 tokens; the maximum prepared and returned usage across the entire suite was 7,910/7,910. No 32k-token forward pass was attempted.

The suite runner wrote results under the constant key `laya-multilingual-mlx` in some JSON structures because that is how the existing scripts label their default backend. This does **not** identify the evaluated model: the HTTP request explicitly selected `liquidai-d1-3b-pytorch-mps`, the adapter rejected mismatches, and its response plus SDK response identified that same alias and the official checkpoint revision. The task-generated `suite_summary.json` records the actual model separately.

## Historical comparison

Same result definitions and hand-written suite assets; historical rows were read, not rerun. Kev-4B and Clef-Flash-8B files are from `origin/main@8e714df8dbf7acffe098e686295a10df74b1d9ef`; Strands Decider v2.1 files are from `eval/strands-decider-v21@8775969b63b3a29ad91b2d77bd311e2e1a341f53`, tag `strandsv21mlx_20261009`.

| Model/result | Single p50 ms | Batch-5 p50 ms | Choice-50 acc | Negated noul acc | Score MAE ↓ | EOU choice AUC | React acc | Memory choice AUC | Drift choice AUC | Wiki link choice AUC | Max successful context usage tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **LiquidAI/d1-3B (this run)** | 49.5 | 127.0 | 0.833 | 1.000 | 0.281 | 0.896 | 0.689 | 0.978 | 1.000 | 0.982 | 7,905 |
| Kev-4B | 47.0 | 149.0 | 1.000 | 1.000 | 0.194 | 0.882 | 0.667 | 1.000 | 1.000 | 0.977 | 7,882 |
| Clef-Flash-8B | 353.9 | 1,231.5 | 0.833 | 1.000 | 0.162 | 0.976 | 0.844 | 1.000 | 1.000 | 0.992 | 8,018 |
| Strands Decider v2.1 | 57.0 | 186.1 | 0.917 | 1.000 | 0.490 | 0.909 | 0.356 | 0.925 | 0.995 | 0.921 | 3,886 |

The M4 Pro/48 GB hardware is shared context for these runs, but quantization, runtime/kernel versions, and run conditions differ; this is not a controlled A/B. The table intentionally separates task accuracy/AUC from confidence and uses the standard EOU `choice2` result. Alternate-wording scores are reported as ranges, not averaged; wording and polarity matter. Strands' context results reached only the ~4k range in its stored result set, so its context figure is not an 8k comparison.

## License, limitations, and operational notes

- The checkpoint's `LICENSE` is **LFM Open License v1.0**, not Apache-2.0. It defines a $10M annual-revenue Threshold and conditions Commercial Use on the Legal Entity not exceeding it; another clause excludes Commercial Use by an entity that exceeds it. Because of this wording at equality, treat an entity at/near the threshold conservatively and request legal/business review. Revenue is evaluated at Legal Entity scope (including controlled/common-control entities); this is not legal advice.
- The Japanese suites are explicitly described in `data.py`, `data_dialog.py`, and `data_wiki.py` as small hand-written smoke-level sets, not benchmark-quality evidence. Text-only suite run: image ability and general Japanese generation/fluency were not tested.
- The maximum measured input was about 7.9k, not the configured 32,768-token boundary; no long-context accuracy beyond the existing suite was tested. Peak MPS memory was not instrumented.
- No recommendation catalog, product backend, upstream model code, remote branch, PR, or shared service was changed. This is a task-only loopback wrapper around the official native API.
- After the full suite, the first task-owned server (PID 15019) was stopped and the port was verified free. After adding the reusable `http_smoke.py`, a second server (PID 19144) was started on the same loopback port solely to execute that script and the SDK smoke; the first PID was already absent. The second server was then stopped. Final `ps`/`lsof` checks found no task server process and no listener on 8025; the two startup notices were sequential restarts, not duplicate live servers.
- Both orderly server shutdowns logged a Python `multiprocessing.resource_tracker` warning about one leaked semaphore; this did not affect inference or suite completion. No child process or listener remained. Revisit if this adapter is used as a persistent service.

## Reproduction and artifacts

Runtime/setup and test details are in [`eval/d1/README.md`](../../eval/d1/README.md); exact direct dependency pins are in [`eval/d1/requirements.txt`](../../eval/d1/requirements.txt). The reusable task-only adapter, native/SDK smoke scripts, fake-runtime contract tests, and count/comparison aggregator are in `eval/d1/`.

| Artifact | Path |
|---|---|
| Main suite aggregate | [`eval/ja/results_liquid_d1_mps_20261010.json`](../../eval/ja/results_liquid_d1_mps_20261010.json) |
| Time-series | [`eval/ja/results_ts_liquid_d1_mps_20261010.json`](../../eval/ja/results_ts_liquid_d1_mps_20261010.json) |
| Dialog | [`eval/ja/results_dialog_liquid_d1_mps_20261010.json`](../../eval/ja/results_dialog_liquid_d1_mps_20261010.json) |
| Dialog wording variants | [`eval/ja/results_dialog_wordings_liquid_d1_mps_20261010.json`](../../eval/ja/results_dialog_wordings_liquid_d1_mps_20261010.json) |
| Synthetic wiki judgments | [`eval/ja/results_wiki_synth_liquid_d1_mps_20261010.json`](../../eval/ja/results_wiki_synth_liquid_d1_mps_20261010.json) |
| Source-derived counts + history comparison | [`eval/d1/suite_summary.json`](../../eval/d1/suite_summary.json) |
| Native inference output | [`eval/d1/native_smoke.json`](../../eval/d1/native_smoke.json) |
| Before/after aggregate adapter counters | [`eval/d1/stats_before.json`](../../eval/d1/stats_before.json), [`eval/d1/stats_after.json`](../../eval/d1/stats_after.json) |

Reproduction commands, from the repository root:

```sh
uv --no-config venv --python 3.12 /Users/yuya/.local/share/systemone-d1-3b-eval
uv --no-config pip install --python /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python -r eval/d1/requirements.txt

env -u PYTHONPATH -u PYTHONHOME HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /Users/yuya/.local/share/systemone-d1-3b-eval/bin/python eval/d1/server.py --host 127.0.0.1 --port 8025

# In another shell after /healthz reports the pinned model and MPS:
PYTHON=/Users/yuya/.local/share/systemone-d1-3b-eval/bin/python \
  ./eval/ja/run_all.sh http://127.0.0.1:8025 liquidai-d1-3b-pytorch-mps liquid_d1_mps_20261010

# Recompute counts/comparisons using the saved live counter snapshots:
/Users/yuya/.local/share/systemone-d1-3b-eval/bin/python eval/d1/aggregate.py \
  --tag liquid_d1_mps_20261010 --before eval/d1/stats_before.json \
  --after eval/d1/stats_after.json --output eval/d1/suite_summary.json
```

Verification on the final task tree: task-local contract tests `4 passed`; repository `uv --no-config sync --extra test` succeeded; `uv --no-config run pytest -q` returned **79 passed with 1 Starlette/AnyIO deprecation warning**. `git diff --check` and final branch/status are recorded with the local commit; nothing was pushed.

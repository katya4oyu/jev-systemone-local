# Clef-Flash MLX 4bit：日本語評価・実測レポート

**実測・suite・比較・cleanup完了。** 固定4bit checkpointを専用worktreeからlocalhostで実推論し、既存日本語評価を全phase一巡した。8021のbind blockerは原因を追い続けず、許可された空きport `8022` へ切り替えて解消した。結果は以下のとおり。小規模・単回の評価であり、一般的な実用性能や統計的有意差を示すものではない。

## モデルprovenance

- Hugging Face repo：`mlx-community/clef-flash-4bit`
- 固定revision：`140bf7e037f5fa95a96535feca112f46c15927cf`
- Base：`Cloudflare/clef-flash`（公式revision `17f0b0ad64efb65d273590632833508766b2aae6`）
- Snapshot：`/Users/yuya/.cache/huggingface/hub/models--mlx-community--clef-flash-4bit/snapshots/140bf7e037f5fa95a96535feca112f46c15927cf`
- `config.json`：`bits=4`, `group_size=64`, `mode=affine`。backboneは9B。model shard 2個とjoint headの合計は `6,193,757,576 bytes`（約6.19 GB）。
- custom `clef_mlx.py` は8bit固定revisionのloaderとSHA-256 `852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5` が一致。推論はjoint schema headを使用し、text generation APIは使っていない。

## 実行環境・接続確認

- Python 3.12.12、arm64 Apple Silicon、RAM `51,539,607,552 bytes`、MLX default device `Device(gpu, 0)`、Metal available。
- 既存隔離環境 `/Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv` を再利用。mlx 0.32.3、mlx-lm 0.32.0、mlx-vlm 0.7.4、transformers 5.18.0、huggingface-hub 1.33.0、httpx 0.28.1、typesafe-sdk 0.7.2。環境の再install/変更はしていない。
- REST `/health` と `/v1/models` は200、返されたmodel idは `clef-flash-4bit`。明示model名でのREST実推論は200、request/responseとも同名、choice/noul/scoreの3回答を返した（377 input tokens、server-reported 860.4 ms）。
- 公式 `typesafe-sdk 0.7.2` の型付きclientでもstatus 200。request/response model名とanswer type 3種を確認した。
- `--no-truncate`で起動。最大長はloader/HTTP sourceの `max_length=16384`。必須over-limit試験では175,142 tokens相当の入力がHTTP 413となり、拒否理由に最大16,384が含まれた。loaderは `truncate=False` 時にstateを切り詰めず `ContextTooLong` をraiseし、HTTP層が413へ変換する。**16,384ちょうどの実入力成功境界は試していない。** smoke集計：`/Users/yuya/.hermes/cache/scratch/clef_flash_4bit_smoke.json`。
- 8021はEADDRINUSEだったため繰り返しbindせず、許可を得た `127.0.0.1:8022` へ切り替えた。他processの停止や8021の操作はしていない。

## 既存日本語suite：すべてexit 0

`eval/ja/run_all.sh` を明示的な `EVAL_MODEL=clef-flash-4bit`, `EVAL_TAG=clefflash4`, `EVAL_BASE=http://127.0.0.1:8022` で実行した。core（6 task）はrun script exit 0、timeseries、dialog、dialog wordings、wiki synth、private wiki、wiki variantsも各exit 0。

| 指標 | Clef-Flash 4bit |
|---|---:|
| latency p50 / p95、単発（client計測） | 354.4 / 358.2 ms |
| latency p50 / p95、5問batch（client計測） | 1,123.4 / 1,172.8 ms |
| 問い合わせ5分類、2 / 5 / 10 / 20 / 50 labels（各n=24） | 0.958 / 0.917 / 0.917 / 0.875 / 0.792 |
| label language accuracy、日本語 / 英語 | 0.917 / 0.917 |
| noul accuracy、positive / negated / English（各n=20） | 0.950 / 1.000 / 1.000 |
| score Pearson r / MAE（n=10） | 0.982 / 0.180 |
| context、約8k tokens、本文が末尾 / 先頭（各n=12） | 0.917 / 0.917（実測input 8,018 tokens、errors 0） |

- **時系列**（各error 0）：trend raw len12/60 = 0.911/1.000、trend summary len12/60 = 1.000/1.000。spike raw len12/60 = 1.000/0.850、spike summary len12/60 = 1.000/0.833。next-up raw/summary = 0.556/0.433（各n=90）。
- **会話**：EOU AUC（noul/choice）0.949/0.978（n=60）、memory 0.990/0.995（n=40）、topic drift 1.000/1.000（n=40）、応答/相槌/無反応accuracy 0.867（n=45）。
- **wordings**：EOUの「話し終えた？」noul AUC/accuracy 0.722/0.550、「文が完結？」0.973/0.850、補正済み「まだ続きがありますか？」0.972/0.883、「言い終わり/言いかけ」choice 0.979/0.900、「完全な文/途切れた文」0.988/0.800。memoryの個人情報/保存すべきかnoulは0.968/0.812、topic-drift各条件AUCは1.000。意味が異なる問いを単なる言い換え耐性とは扱わない。
- **合成wiki**（link/split/duplicate）：noul/choice AUCはlink 0.993/0.991、split 0.983/1.000、duplicate 1.000/1.000。
- **私的wiki**（421 knowledge / 50 memos）：link AUC 0.990（n=298）、sense AUC 0.872（n=200）、knowledge種別accuracy 0.61（n=100、majority baseline 0.96）、memos種別accuracy 0.94（n=50、baseline 0.66）、errors 0。Frontmatterを機械goldとした小規模評価なので、特にknowledge種別はmajority baselineを下回る。
- **wiki variants**（各形式160組）：AUC / accuracy@0.5 はnoul full 0.975/0.875、noul short 0.999/0.944、choice pair 0.998/0.850、choice+context 0.988/0.831、score relatedness 0.963/0.850。
- 極性補正済みfocused check `run_dialog_wording_polarity_check.py` はexit 0：n=60、accuracy@0.5 0.883、AUC 0.972、mean probability 0.908（finished）/0.312（not finished）。既存 `noul_for_finished()` を用い、従来のraw評価極性誤りと混同しない別JSON `results_dialog_wordings_eou_continue_fixed_clefflash4.json` に保存した。

## compare.py：4bit / 8bit / historical Kev-4B

`eval/ja/compare.py` exit 0。出力表には `clef-flash-8bit` と `clef-flash-4bit` の両列が入り、Kev-4B列は既存記録から取得した。下表は同出力からの抜粋（精度/AUCはcompare.py表示の2桁丸め）。Kev-4Bはhistorical baselineで、今回再測定していない。

| 指標 | Kev-4B既存値 | Clef-Flash 8bit | Clef-Flash 4bit |
|---|---:|---:|---:|
| 5分類、5 labels accuracy（n=24） | 1.00 | 0.96 | 0.92 |
| 5分類、50 labels accuracy（n=24） | 1.00 | 0.83 | 0.79 |
| positive / negated noul accuracy（各n=20） | 1.00 / 1.00 | 1.00 / 1.00 | 0.95 / 1.00 |
| score Pearson r（n=10） | 0.98 | 0.98 | 0.98 |
| context約8k、本文末尾 accuracy（n=12） | 1.00 | 0.92 | 0.92 |
| EOU choice AUC（n=60） | 0.88 | 0.98 | 0.98 |
| 実在wiki link AUC（n=298） | 0.95 | 0.99 | 0.99 |
| latency p50、単発 / 5問batch | 47.00 / 149.00 ms | 353.90 / 1,231.50 ms | 354.40 / 1,123.40 ms |

極性補正済みEOU focused 60件は4bit accuracy 0.883 / AUC 0.972、8bit accuracy 0.883 / AUC 0.987。単発遅延はこの測定でほぼ同じ、5問batch p50は4bitの方が短かった。数値は一回のローカル測定であり、環境差のある歴史的Kev値との性能保証や統計的有意差を意味しない。

## 出力・プライバシー検査

- 期待した評価JSON 8個（core、timeseries、dialog、wordings raw、wordings corrected、wiki synth、wiki、wiki variants）は8/8 JSON parse成功。
- 37個のerror/error-count fieldはすべて0。privacy scanで `state`, `questions`, `body`, `snippet` 等のprivate payload key 0、512文字を超えるstring 0。wiki本文はlocalhost推論のみで扱い、保存したのは集計JSONだけ。
- 評価結果JSON：`eval/ja/results_clefflash4.json`, `results_ts_clefflash4.json`, `results_dialog_clefflash4.json`, `results_dialog_wordings_clefflash4.json`, `results_dialog_wordings_eou_continue_fixed_clefflash4.json`, `results_wiki_synth_clefflash4.json`, `results_wiki_clefflash4.json`, `results_wiki_variants_clefflash4.json`。

## 再現手順

```bash
SNAPSHOT=/Users/yuya/.cache/huggingface/hub/models--mlx-community--clef-flash-4bit/snapshots/140bf7e037f5fa95a96535feca112f46c15927cf
env -u PYTHONPATH -u PYTHONHOME \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I \
  "$SNAPSHOT/clef_mlx.py" serve --model "$SNAPSHOT" \
  --name clef-flash-4bit --host 127.0.0.1 --port 8022 --no-truncate
```

```bash
cd /Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/clef-flash-4bit
env -u PYTHONPATH -u PYTHONHOME \
  EVAL_BASE=http://127.0.0.1:8022 EVAL_MODEL=clef-flash-4bit EVAL_TAG=clefflash4 \
  PYTHON="/Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python /Users/yuya/.hermes/cache/scratch/clef_flash_4bit_phase_wrapper.py" \
  ./eval/ja/run_all.sh http://127.0.0.1:8022 clef-flash-4bit clefflash4 \
  /Users/yuya/src/github.com/katya4oyu/me/notes

env -u PYTHONPATH -u PYTHONHOME \
  EVAL_BASE=http://127.0.0.1:8022 EVAL_MODEL=clef-flash-4bit EVAL_TAG=clefflash4 \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I \
  eval/ja/run_dialog_wording_polarity_check.py
```

## 検証・cleanup

- append-only log：`/Users/yuya/.hermes/cache/scratch/clef_flash_4bit_eval.log`。同一inode `130388536` を維持。
- `wiki_variants` は04:29:09にexit 0、focused checkは08:58:40に開始し08:59:04にexit 0。phase logにこの間の時間差の理由は記録されていないため、推論時間とはみなさない（原因は未確認）。
- notification lifetime capに達したcollector PID 60857はtask-ownedのsentinelでexit 0を確認し、同じscriptをPID 94181で再起動。readiness後の残phase通知を受け取り、全phase後にsentinelでexit 0を確認。
- task-owned model server PID 60841（固定snapshot、`127.0.0.1:8022`, `--no-truncate`）は評価後に停止。PID 60841/94181/61032はすべて終了し、8022 listenerなし、health probeはconnection refused。8021は操作していない。
- `env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q` はこのレポート反映後にexit 0、65 passed / 0 failed。Starlette/anyio DeprecationWarning 1件。`git diff --check` もexit 0。

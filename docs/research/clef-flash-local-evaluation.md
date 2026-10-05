# Clef-Flash 8bit MLX: 日本語評価・実測レポート

**状態: 実測・評価・ローカルrepo checks完了。** 2026-10-04の一回の実行でモデル取得、ロード、REST/SDK疎通、既存日本語suite、採点訂正のfocused評価を実施。結果と制約を以下に記録する。

## モデル取得と実行確認

- Hugging Face: `mlx-community/clef-flash-8bit`、revision `dfa0993decb4f8507a0eae01afd1b2d33a4bb734`
- Snapshot: `/Users/yuya/.cache/huggingface/hub/models--mlx-community--clef-flash-8bit/snapshots/dfa0993decb4f8507a0eae01afd1b2d33a4bb734`
- 重み2 shardとjoint headの合計: 10,670,128,927 bytes（10.670 GB）。8bitの9Bモデルであり、4bit版や27B版への置換ではない。
- Custom loader `clef_mlx.py` の入力エンコード、joint schema head、推論、context制限、HTTP serve部を読んでから実行。`mlx-vlm`経由のbackboneと同梱headで動く。テキスト生成APIは使っていない。
- Python 3.12.12のtask専用環境: mlx 0.32.3、mlx-lm 0.32.0、mlx-vlm 0.7.4、transformers 5.18.0、huggingface-hub 1.33.0、httpx 0.28.1、typesafe-sdk 0.7.2（uv 0.12.0）。hostはarm64 Apple Silicon、`hw.memsize=51539607552` bytes。repo revisionやHermes環境は変更していない。
- 127.0.0.1:8020でserve。明示model名`clef-flash`。起動引数に`--no-truncate`（最大16,384 token）を指定。
- `GET /health`と`GET /v1/models`はいずれも200、model idは`clef-flash`。明示modelを指定した実推論は200でchoice/noul/scoreの3回答を返した（340 input tokens、server-reported 821.9 ms）。
- 75,139 token相当のoversize入力は413となり、16,384上限を越えたstateが切り詰められず拒否されることを確認。
- 公式Python SDK `typesafe-sdk 0.7.2`からもローカルmodelを明示した実推論成功。回答3種をSDKの型付き結果として受け取った。
- smoke集計: `/Users/yuya/.hermes/cache/scratch/clef_flash_smoke.json`。送信した短い合成入力のみで、ユーザーwiki本文は含まない。

## 既存日本語suiteの実測

`eval/ja/run_all.sh`はexit 0（latency、choice、label-language、noul、score、context、時系列、会話、wordings、合成wiki、実wiki、wiki variants）。結果は専用tagのJSONへ保存。既存suiteの一部JSON keyはコード上`laya-multilingual-mlx`のままだが、全リクエストのmodel overrideは`clef-flash`。実推論応答にもmodel名を確認した。

| 指標 | Clef-Flash 8bit 実測 |
|---|---:|
| client計測 latency p50 / p95、単発 | 353.9 / 357.4 ms |
| client計測 latency p50 / p95、5問batch | 1,231.5 / 1,256.1 ms |
| 問い合わせ5分類、2 / 5 / 10 / 20 / 50ラベル | 0.958 / 0.958 / 0.958 / 0.917 / 0.833（各24件、error 0） |
| ラベル表記、日本語 / 英語 | 0.958 / 0.958 |
| noul、肯定 / 否定 / 英語質問 | 1.000 / 1.000 / 1.000（各20件、error 0） |
| score、Pearson r / MAE | 0.984 / 0.162（10件） |
| context、約8,000 input tokens、本文が末尾 / 先頭 | 0.917 / 1.000（各12件、error 0） |

## 既存Kev-4Bとの比較（`compare.py`実出力）

| 指標 | Kev-4Bの既存実測 | Clef-Flash 8bit今回 |
|---|---:|---:|
| 5分類・5ラベル accuracy（24件） | 1.00 | 0.958 |
| 5分類・50ラベル accuracy（24件） | 1.00 | 0.833 |
| 否定形toxicity noul（20件） | 1.00 | 1.000 |
| score Pearson r（10件） | 0.98 | 0.984 |
| 約8k token、本文が末尾（12件） | 1.00 | 0.917 |
| 発話区切り choice AUC（60件） | 0.88 | 0.976 |
| 実wiki長文リンク AUC（298件） | 0.95 | 0.992 |
| latency p50・単発（client計測） | 47.0 ms | 353.9 ms |

5ラベル分類、否定形noul、score、会話ターンとリンク判定では今回の値はKev-4Bに近いか同等だが、50ラベル分類と長文末尾の精度は下回り、単発遅延は大きい。どちらも少数の自作/私的wikiデータによる単回測定であり、性能保証や統計的有意差を意味しない。Kev-4B値は既存記録をそのまま参照し、今回再測定していない。

## 時系列・会話suite（完了）

- `run_ts.py` exit 0。trendはraw 12点0.889/60点1.000、summaryは12点/60点とも1.000（各n=90）。spikeはraw 12点1.000/60点0.833、summary 12点1.000/60点0.833（各n=60）。次値上昇の判定はraw 0.567、summary 0.433（各n=90）。
- `run_dialog.py` exit 0。発話区切りAUCはnoul 0.933、choice 0.976（各n=60）。途中切断を含むセットはnoul 0.876、choice 0.948（各n=44）。応答/相槌/無反応 choice accuracy 0.844（n=45）。記憶判定AUCはnoul 0.988、choice 1.000（各n=40）。話題逸脱はnoul/choiceともAUC 1.000、accuracy 1.000（各n=40）。
- `run.py`のlatency・choice・label-language・noul・score・context各条件はすべてerror 0。contextでは約6k/8k input tokensを各12件、約300〜4kを各24件実行した。

## Wording・合成wikiの途中結果

- `run_dialog_wordings.py` exit 0。話者が話し終えたかの表現では、「話し終えた？」accuracy 0.583/AUC 0.746、「文として完結？」0.900/0.981、「言い終わり/言いかけ」choice 0.917/0.977、「完全な文/途切れた文」0.883/0.991。質問の意味が異なるものもあり、単純な言い換え耐性とはまとめない。
- `run_dialog_wordings.py`の「まだ続きがありますか？」は、`EOU_ITEMS`のgold（話し終えた）と極性が逆なのに、確率を反転せず採点していた。raw JSONのaccuracy 0.133/AUC 0.013は**評価側の極性不整合で無効**で、Clef-Flashの弱さを表す値ではない。raw JSONと他モデルの履歴は保持したまま、`noul_for_finished()`の最小修正を追加した。補正済み60件の実モデル再評価はaccuracy 0.883、AUC 0.987、mean probability 0.900（finished）/0.238（not finished）、error 0。別JSON: `eval/ja/results_dialog_wordings_eou_continue_fixed_clefflash8.json`。
- 訂正用単体テストはRED（補正関数が未実装）を確認後GREEN。既存の全wordings raw結果には改変を加えていない。
- 記憶項目では「個人情報か」noul accuracy 0.925/AUC 0.978に対し、「後で思い出すため保存すべきか」0.625/0.863。これは意味の異なる判断であり、単なる言い回し差として扱わない。話題逸脱は反転noul AUC 1.000、同話題noul 1.000、choice 1.000。
- `run_wiki_synth.py` exit 0。link AUC 0.992/0.992（noul/choice）、split 0.980/1.000、duplicate 1.000/1.000。
- `run_wiki.py`（実在の私的wiki）はexit 0。評価対象421 knowledge/50 memosから、リンク妥当性AUC 0.992（298件）、sense判定0.856（200件）、knowledge種別accuracy 0.61（100件）、memos種別0.96（50件）。knowledge型セットはconceptが96/100でmajority baseline 0.96、memosはcaptureが33/50でbaseline 0.66。Frontmatterラベルを機械的なgoldとして使ったため、特にknowledge型accuracyは多数派・ラベル妥当性の制約込みで読み、能力保証としない。処理はlocalhostのみ、保存結果は集計値のみ。
- `run_wiki_variants.py` exit 0（各形式160組）。AUC / accuracy@0.5はnoul full 0.985/0.912、noul short 0.999/0.950、choice pair 0.996/0.863、choice pair+context 0.988/0.844、score relatedness 0.977/0.844。

## 最終確認

- `eval/ja/compare.py`を通して既存比較表へ`clef-flash-8bit`列を追加し、表の生成exit 0を確認。
- 8個の評価JSONはすべてparse成功、非0のerror件数なし、私的本文を示すstate/questions/body/snippetフィールドなし。focused EOU結果はn=60、accuracy 0.883、AUC 0.987。
- JSON: `eval/ja/results_clefflash8.json`, `results_ts_clefflash8.json`, `results_dialog_clefflash8.json`, `results_dialog_wordings_clefflash8.json`（問題のraw metricを含むが上記のとおり無効と明記）, `results_dialog_wordings_eou_continue_fixed_clefflash8.json`, `results_wiki_synth_clefflash8.json`, `results_wiki_clefflash8.json`, `results_wiki_variants_clefflash8.json`。
- `uv --no-config sync --extra test` exit 0。`env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q` は65 passed、1件のStarlette deprecation warning。
- polarity単体テストは未実装状態でREDを確認し、修正後GREEN。全pytestにも含まれpass。
- 推論用task-owned server PID 22725は評価完了後に意図して停止。port 8020 listenerなし、PID終了、`/health` probeは接続失敗（exit 7）を確認。
- 実行phase log: `/Users/yuya/.hermes/cache/scratch/clef_flash_eval.log`。モデル重みと一時環境はscratch/HF cacheにあり、worktreeには追加していない。


## 再現

専用Python環境と`PYTHONPATH`/`PYTHONHOME`未設定が必要（Hermesの別Python site-packages混入を避ける）。最初に固定snapshotからlocalhost限定、入力切り詰めなしで起動する。

```bash
SNAPSHOT=/Users/yuya/.cache/huggingface/hub/models--mlx-community--clef-flash-8bit/snapshots/dfa0993decb4f8507a0eae01afd1b2d33a4bb734
env -u PYTHONPATH -u PYTHONHOME \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python \
  "$SNAPSHOT/clef_mlx.py" serve --host 127.0.0.1 --port 8020 --name clef-flash --no-truncate
```

別terminalで既存全suiteを実行する。第4引数は省略可能なlocal wiki path。

```bash
cd /Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/clef-flash-8bit
env -u PYTHONPATH -u PYTHONHOME \
  EVAL_BASE=http://127.0.0.1:8020 EVAL_MODEL=clef-flash EVAL_TAG=clefflash8 \
  PYTHON=/Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python \
  ./eval/ja/run_all.sh http://127.0.0.1:8020 clef-flash clefflash8
```

採点極性を補正したEOUのfocused 60件だけを別JSONへ再現する。

```bash
cd /Users/yuya/src/github.com/katya4oyu/systemone-workbench/.worktrees/clef-flash-8bit/eval/ja
env -u PYTHONPATH -u PYTHONHOME \
  EVAL_BASE=http://127.0.0.1:8020 EVAL_MODEL=clef-flash EVAL_TAG=clefflash8 \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python \
  run_dialog_wording_polarity_check.py
```

`uv --no-config sync --extra test`と`env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q`がrepo testsの実行条件。実測時のphase logは`/Users/yuya/.hermes/cache/scratch/clef_flash_eval.log`。

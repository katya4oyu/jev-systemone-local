# JAFFEでのClef-Flash MLX画像評価

**対象:** 既存のClef-Flash 9B視覚モデル 4bit / 8bitを使った、ローカル限定の科学的評価。Webデモ、画像アップロード機能、production backend変更はこの作業に含めない。

## 結果

両モデルとも**213/213画像を実推論**し、分類誤り扱いとなる実行／回答エラーは0件。212枚の評定一致画像では6次元すべて有効な予測を返した。accuracyの分母は推論エラーを含む213枚。

### 7択ポーズ分類

| 固定checkpoint | accuracy (n=213) | macro-F1 | 多数派baseline | errors |
|---|---:|---:|---:|---:|
| Clef-Flash 4bit | 0.431925 | 0.379193 | 0.150235 | 0 |
| Clef-Flash 8bit | 0.436620 | 0.389063 | 0.150235 | 0 |

| 配布ポーズ | n | 4bit accuracy | 8bit accuracy |
|---|---:|---:|---:|
| NE / 無表情 | 30 | 1.000000 | 1.000000 |
| HA / 喜び | 31 | 0.645161 | 0.645161 |
| SA / 悲しみ | 31 | 0.322581 | 0.322581 |
| SU / 驚き | 30 | 0.866667 | 0.866667 |
| AN / 怒り | 30 | 0.000000 | 0.000000 |
| DI / 嫌悪 | 29 | 0.000000 | 0.034483 |
| FE / 恐れ | 32 | 0.187500 | 0.187500 |

4bit / 8bitはこの標本で僅差。どちらも多数派baselineを上回る一方、怒りは両モデルで正答なし。4bitは嫌悪も正答なしで、neutral予測に偏る。詳細な集計confusion matrix、per-class precision/recall/F1（予測数0のprecisionは`null`）は各JSONを参照。これは依頼されたポーズ分類であり、感情認識の精度ではない。

### 人の段階評定との比較

モデルscoreの期待値0〜4に1を足して、人のREADME平均評定1〜5と照合。各次元n=212、全次元をまとめたn=1,272ペア。MAEは低いほど近く、Pearson rは同じ画像間での線形な傾向一致。

| 評定 | 人平均 | 4bit平均 | 4bit MAE / r | 8bit平均 | 8bit MAE / r |
|---|---:|---:|---:|---:|---:|
| HAP / 喜び | 2.199387 | 1.398033 | 0.806154 / 0.813757 | 1.437295 | 0.766247 / 0.811063 |
| SAD / 悲しみ | 2.589151 | 1.866646 | 0.834980 / 0.671266 | 1.927925 | 0.788870 / 0.666195 |
| SUR / 驚き | 2.450755 | 1.806239 | 0.659934 / 0.892548 | 1.866972 | 0.602309 / 0.893715 |
| ANG / 怒り | 2.465896 | 1.304068 | 1.162293 / 0.640028 | 1.395170 | 1.072690 / 0.672031 |
| DIS / 嫌悪 | 2.768302 | 1.419232 | 1.356311 / 0.725959 | 1.445866 | 1.333226 / 0.704272 |
| FEA / 恐れ | 2.338915 | 1.647340 | 0.757739 / 0.640845 | 1.719704 | 0.715435 / 0.621056 |
| 全6次元 | — | — | 0.929568 / 0.669462 | — | 0.879796 / 0.667576 |

両quantizationとも人平均よりモデル平均が全6次元で低く、特に嫌悪・怒りでMAEが大きい。6次元を合算したPearsonはどちらも約0.67で、8bitで一様に改善する傾向でもない。小規模の記述値であり、有意差・一般性能を示すものではない。

### Vision path・速度

各画像の固定7問を1 requestにまとめた。両モデルとも全213件の`pixel_values` shapeは`256×1536`、`image_grid_thw` shapeは`1×3`、入力token数は全件1,203（合計256,239）。4bitは別のencoding-only全件走査でも同じ値を確認。実画像2件と画像なし対照のsmokeでは、両モデルとも画像間で7回答（choice＋6 score）が変化し、happyポーズ画像と画像なしのchoiceも異なった。画像なしは精度に含めない。

| 計測 | 4bit | 8bit |
|---|---:|---:|
| Full suite request区間合計 | 601.080秒 | 609.015秒 |
| Full suite p50 / p95 | 2,815.319 / 2,904.542 ms | 2,850.586 / 2,988.229 ms |
| 5 warm-up後の40回 p50 / p95 | 2,884.277 / 3,077.247 ms | 2,933.230 / 3,109.485 ms |

時間は既にロード済みモデルへのMLX画像processor＋vision/backbone＋joint headを対象とし、モデルロードとTIFF読み込み／PIL変換を含まない。Full suite合計は213 request区間の合計で、end-to-endの画像I/O時間ではない。既存のテキスト評価の速度と直接比較しない。

結果JSON: [`4bit`](../../eval/vision/results/clef-flash-4bit.json)、[`8bit`](../../eval/vision/results/clef-flash-8bit.json)。


## データと利用条件

- 公式配布元: [Zenodo JAFFE record 14974867](https://zenodo.org/records/14974867), DOI [10.5281/zenodo.14974867](https://doi.org/10.5281/zenodo.14974867)。取得した`jaffe.zip`は12,290,558 bytes、Zenodo APIのMD5 `fe13f3302eb9968ef04367456f665436`と一致。`README_FIRST.txt`は18,459 bytes、APIのMD5 `3d3e20343692832266cb094f0c54c5c0`と一致。データ本体はリポジトリ外の`~/.hermes/cache/scratch/jaffe-official-14974867/`に置き、追跡・再配布しない。
- 画像213枚。すべて256×256、8-bit TIFF。Pillowで読み取ったmodeは`L`が190枚、グレースケールpaletteの`P`が23枚（全palette entryでR=G=Bを確認）。入力時は全て`L`へ変換した後にQwen3-VL processor用のRGB PIL画像へ変換した。画像表示・外部送信はしない。
- Zenodoの利用条件は**非商用の科学研究のみ**。GitHub等へのデータ再配布、Web/SNSへの画像掲載、公衆展示、マスメディア放送は禁止。許諾はデータセット用であり、本評価の集計結果を画像データの再配布許可とは扱わない。画像や一画像ごとの予測・ラベルは成果JSONへ保存しない。
- 配布READMEは7つの依頼ポーズ（NE/HA/SA/SU/AN/DI/FE）と、6つの形容表現の段階評定（HAP/SAD/SUR/ANG/DIS/FEA、1〜5）を説明する。主要な6次元評定表は219行。ファイル名のポーズIDと**厳密一致する画像は212枚**で、画像側だけのIDは`YM-HA2`、評定表側だけのIDは`KM-DI2`, `KM-HA5`, `KM-SA4`, `KR-HA3`, `NM-DI2`, `TM-HA4`, `YN-HA2`。元データの`YN-HA2`を`YM-HA2`へ黙って直さず、相互に結び付けない。画像ポーズ分類は全213枚、評定比較は一致した212枚だけを分母とする。
- 恐怖刺激と恐怖評定を除いた別実験の5次元表（187行）は6次元表へ混ぜず、今回使わない。KASRLの紹介ページは観察者62名と記す一方、配布READMEは主要表を60名の日本人女性学生の平均と記す。評定データの出典・人数はREADMEの60名表記を採用し、差を隠さない。

## 研究上の解釈と利用境界

必須引用の2論文を本文まで確認した。

1. Lyons, Kamachi, Gyoba, [“Coding Facial Expressions with Gabor Wavelets (IVC Special Issue)”](https://arxiv.org/abs/2009.05938) (arXiv:2009.05938). 元研究は顔画像の類似性表現を人の意味評定との関係で検討し、ポーズラベルをそのまま正解分類として使うことを目的にしていない。本文は歴史的な実験について219枚・92人（6評定のfearあり2群31人ずつ、別実験のfearなし5評定2群15人ずつ）と記す。一方、後年配布READMEは213画像と主要評定表60人を説明する。これらの異なる時点・実験の数を同一の出典として混同しない。
2. Lyons, [“‘Excavating AI’ Re-excavated: Debunking a Fallacious Account of the JAFFE Dataset”](https://arxiv.org/abs/2107.13998) (arXiv:2107.13998). 著者はJAFFEを人の顔表情知覚研究のための、依頼してポーズを取ってもらった刺激と説明し、訓練用機械学習データや内面の感情ラベルとして作ったものではないと論じる。また文化差、ポーズと観察者評定の不一致、実世界への一般化の限界を説明する。

したがって本評価では「人物が本当に感じている感情」「性格」「個人同定」を推測せず、画像に**見える表情**だけを扱う。7択のgoldはファイル名が表す依頼ポーズであり、人が見た主観的評定とは別の対象である。6つのscoreは各画像について公開READMEにある人の平均評定との傾向比較であって、内面状態の正解ではない。

## 固定した評価手順

- 各画像に対して同一の日本語stateと7問（表情7択1問＋6次元の1〜5段階score）を1 requestにまとめて一度推論する。すべての画像で同じ質問・選択肢・順序を使い、画像名、人物コード、ポーズコード、評定値はstateに含めない。質問構造は実装`eval/vision/jaffe.py::make_questions` / `make_record`に固定。
- 入力はPillow画像のみ。意図的な生成・外部API・HTTP serverは使わない。固定checkpointを`clef_mlx.load(..., backend="vlm")`で読み、`record["images"]`経由のmlx-vlm image processor、vision tower、Clef joint-schema headを通す。テキスト生成APIではない。`truncate=False`、上限16,384 tokens。
- 4bitを単独プロセスで完了・終了してから8bitを実行。推論中に他モデルと競合させない。各runの先頭で同一条件のポーズ画像2枚と画像なしの対照1件を確認する。画像なし対照は精度の分母に含めない。さらに同じ画像・質問によるwarm-up 5回後、40回の単画像レイテンシを測る。
- 分類accuracyは全213枚を分母とし、推論失敗・不正回答も誤りとして数える。macro-F1と7×7 confusion matrix、依頼ポーズ別accuracyを集計する。6次元評定はモデルの期待scoreが0〜4なので+1し、README平均1〜5と比較する。次元ごとのMAE / Pearson rと有効数を出す。いずれかの配列が定数ならPearsonは`null`、0件のprecision等は`null`とする。
- JSONは集計値、固定モデル・データprovenance、エラー数のみ。画像、base64、pixel配列、prompt payload、一画像ごとの予測は保存しない。

## モデル・実行環境

- 4bit: `mlx-community/clef-flash-4bit`, revision `140bf7e037f5fa95a96535feca112f46c15927cf`。
- 8bit: `mlx-community/clef-flash-8bit`, revision `dfa0993decb4f8507a0eae01afd1b2d33a4bb734`。
- いずれも既存の9B vision + joint-schema head snapshotを使用。`clef_mlx.py`のSHA-256は両方とも`852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5`。モデル再取得、環境再構築、微調整は行わない。
- 隔離Python: `/Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I`。実行時パッケージ版とデバイス情報は各JSONに記録する。macOS 27、Apple Silicon arm64、物理メモリ51,539,607,552 bytes。実行はlocalhostやサーバに依存しない直接推論。
- 推論レイテンシの計測範囲は、既にメモリへロードしたモデルへの1画像についてのmlx-vlm image processor、vision/backbone、joint headの壁時計時間。TIFFのディスク読み込み／PILのL→RGB変換とモデルロードは含まない。全suite時間は各requestの同じ区間の合計で、エンドツーエンドの画像I/O時間ではない。

## 再現

公式archiveと`README_FIRST.txt`を上記scratch locationへ置いた後、worktree rootで次を実行する。取得 helperは今回のscratch script `/Users/yuya/.hermes/cache/scratch/jaffe-clef-download.py`（実行: `env -u PYTHONPATH -u PYTHONHOME /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I /Users/yuya/.hermes/cache/scratch/jaffe-clef-download.py`）で、公式Zenodo APIのarchive size/MD5を検証し、ZIP path traversal / symlinkを拒否してscratch内だけに展開した。helperと画像データはこのworktree/Gitへ含めない。初期出力は`eval/vision/results/`、phase logはscratchに書かれる。

```bash
env -u PYTHONPATH -u PYTHONHOME \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I \
  eval/vision/jaffe.py --bits 4
# 4bitプロセス終了を確認してから別プロセスで実行
env -u PYTHONPATH -u PYTHONHOME \
  /Users/yuya/.hermes/cache/scratch/clef-flash-8bit-venv/bin/python -I \
  eval/vision/jaffe.py --bits 8
```

テスト: `env -u PYTHONPATH -u PYTHONHOME uv --no-config run pytest -q`。

## 制約

- 10人の日本人女性による posed なグレースケール静止画、閉じた7ポーズの小規模データ。自然な場面、多様な人口集団、動画、一般的な「感情認識」性能へ一般化できない。JAFFEがモデルのpretrainingに含まれたかは確認できず、汚染の可能性は未評価。
- 小規模な記述的比較であり、信頼区間、有意差検定、精度保証とはしない。人の平均評定も画像の見え方についての主観指標として扱う。
- Zenodoの規約により画像をWebデモや公開レポートへ掲載しない。次のphaseの任意画像Webデモはこの評価の範囲外であり、別途利用条件・設計を確認する。

## 出典

- [Zenodo: The Japanese Female Facial Expression (JAFFE) Dataset, record 14974867](https://zenodo.org/records/14974867) — terms, archive, README_FIRST.txt.
- [KASRL: JAFFE Dataset](https://www.kasrl.org/jaffe.html) — 配布者のdataset説明（62 observers表記）。
- Lyons, Kamachi, Gyoba, [Coding Facial Expressions with Gabor Wavelets](https://arxiv.org/abs/2009.05938), arXiv:2009.05938 (2020).
- Lyons, [“Excavating AI” Re-excavated](https://arxiv.org/abs/2107.13998), arXiv:2107.13998 (2021).
- [mlx-community/clef-flash-4bit](https://huggingface.co/mlx-community/clef-flash-4bit) and [mlx-community/clef-flash-8bit](https://huggingface.co/mlx-community/clef-flash-8bit) — local model cards and custom loader.

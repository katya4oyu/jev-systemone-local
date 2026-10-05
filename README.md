# systemone-workbench

Apple SiliconのMac上で、System Oneモデル（高速に型付きの判断を返す小型モデル）を提供・比較・推薦・評価・微調整するための作業台。

System Oneモデルは、選択肢から選ぶ（`choice`）、段階で評価する（`score`）、はい／いいえを確率で返す（`noul`）判断を、生成なしの1回の推論で返す。TypeSafeのJev、Laya、Jeff、Kevなどがこの種類にあたる。

- **提供**：LayaをMLX・Core MLで動かし、Jev互換の`/v1/systemone`として公開する（公式SDKからそのまま使える）。微調整したチェックポイントも登録できる
- **転送**：別プロセスのJeff・Kevなど、Jev互換サーバを`model`名で使えるようにする
- **推薦**：やりたいことを文章で送ると、どのモデルを使うべきかを返す（`/v1/recommend`）
- **評価**：日本語の自作データで、Jev・Laya・Jeff・Kevを同じ条件で比べる（`eval/ja/`）
- **微調整**：Layaの公式手順をApple Siliconで動かす（`eval/finetune/`）
- **調査**：結果と根拠は [調査メモ](docs/research/jev-vs-laya-case-studies.md) にまとめている

動作はApple SiliconのMacが前提。Jev本体（TypeSafe）とは無関係の独立したプロジェクトで、Jevの重みは含まない。旧名は`jev-systemone-local`で、旧コマンド名`jev-systemone-local`と、旧環境変数`JEV_LOCAL_MLX_MODELS`・`JEV_LOCAL_PROXY_MODELS`も当面は使える（新旧の値が食い違うとエラーにする）。

## 方針

JevのSystem One互換エンドポイントを、ローカルで動かせる実装から提供する。

バックエンドは汎用名ではなく、実際に対応する実装名で管理する。

- `laya-mlx`
- `laya-coreml`
- `laya-onnx`

対応していない実装を、対応済みのように扱わない。

## 実装順

次の順序で対応する。

1. `laya-mlx`
2. `laya-coreml`
3. `laya-onnx`

まず `laya-mlx` でSystem One互換サーバとして利用できる状態を作り、その後に他のバックエンドを追加する。

## サーバ実装

Pythonを使い、依存関係と実行環境の管理には `uv`、HTTP APIには `FastAPI` を使う方針とする。

APIサーバの構造や互換性の扱いは、先行実装の `laya-serve` を参考にする。ただし、対応バックエンドとAPI契約はこのリポジトリで明示的に管理する。

## 起動とiPhoneでの確認

Apple SiliconのMacとPython 3.11以降を使う。初回はモデルの重みをHugging Faceから取得する。

```bash
uv sync --extra test
uv run pytest -q
```

Macだけで試すなら `uv run systemone` で起動し、`http://127.0.0.1:8017/` を開く。iPhoneからTailscale経由で試すときはMacのTailscale IPv4アドレスだけで待ち受ける：

```bash
tailscale ip -4
uv run systemone --host <MacのTailscale IPv4> --port 8017
```

iPhoneも同じtailnetへ接続し、`http://<MacのTailscale IPv4>:8017/` をSafariで開く。画面は同一サーバから配信するTypeSafe公式JavaScript SDK v0.6.0を使い、`models.list()` と `systemOne()` でMLXまたはCore MLの判定を呼び出す。ブラウザ用にはローカルサーバが検証しないダミーキーだけを指定する。実際のAPIキーをブラウザへ渡さない。同じサーバの `/snake` は独自のゲームAPIを使うため、SDK互換性の確認対象ではない。サーバにアプリ独自の認証はないため、公開ネットワークへはバインドしない。ブラウザとAPIの通信はHTTPであり、秘匿すべき文章を入力しない。

## Snakeデモ

`/snake` ではモデル、「開始」「一時停止」「新しいゲーム」、速度の上限を操作できる。盤面、方向ごとのモデル確率、モデルの第一候補、実行した方向、安全補助の介入、推論時間を表示する。各手の判定にはゲーム開始時に選んだモデルを使う。ゲーム盤面は `laya-mlx` 同梱の `SnakeGame`、各モデルの推論には対応する `LayaPolicy` を利用する。文章判定APIとSnakeでモデルを共有し、重みを二重にロードしない。

安全補助は元デモの既定どおり有効。元デモと同様にモデルは安全性や餌への進路の説明を入力として受けるので、盤面だけから戦略を学習した例ではない。画面に表示する確率はモデルの実出力で、移動後の盤面に添える「直前の判断」として扱う。推定値は校正された死亡確率ではない。ゲームはサーバのメモリに最大16件保持し、1時間操作がなければ期限切れになる。サーバ再起動でも失われる。速度の数値は上限であり、推論や通信が遅ければそれ以下になる。

## Clef-Flash画像デモ

`--vision-checkpoint`を指定すると、通常のLaya/Core ML/Snakeアプリとは独立した画像デモを起動する。明示したローカルディレクトリだけを読み、Hugging Faceから重みを取得しない。Macのlocalhostから試す場合：

```bash
uv sync --extra vision
uv run systemone --vision-checkpoint "$HOME/.cache/huggingface/hub/models--mlx-community--clef-flash-4bit/snapshots/140bf7e037f5fa95a96535feca112f46c15927cf"
```

ブラウザで `http://127.0.0.1:8017/` を開く。既存の起動方法は変わらず、フラグなしではこれまでどおりのアプリを起動する。8-bitを使う場合は、手元にある8-bitスナップショットのローカルパスを指定する。起動時に選択したチェックポイントをロードできない場合、`/healthz`がreadyを返す前に起動が失敗する。

JPEG・PNG・WebPを1枚選び、プレビューを見てから判定する。画像は1枚4 MiB以下・1,600万画素以下、推論用に最大2,048 pxへ縮小する。静止画のみ。サーバへは同一オリジンの `/vision/api/analyze` に1リクエストを送り、ローカルMLX画像処理とClef-Flashの推論結果を表示する。これは独自APIであり、画像を扱わない既存の公式SDK `/v1/systemone` 契約は変更しない。画像本体、ファイル名、EXIF情報は保存・ログ出力せず、任意URLや外部APIへは送らない。サーバに認証はないため、既定のlocalhostか、明示したtailnet内のIPだけで待ち受ける。

結果はJAFFE評価と同じ7質問（表情の7択＋6成分の1〜5評価）に、顔が1つはっきり見えるかを確認する補助質問を1つ加えたもの。補助質問は顔検出保証ではなく、判別困難時は7表情分類を表示しない。JAFFEの管理された顔画像での分類正答率は4-bitで43.2%、8-bitで43.7%。任意画像や顔の有無を判定する精度は未評価で、表示確率は本人の実際の気持ちの確率ではない。

「カメラで連続判定」を押してカメラの使用を許可すると、ライブ映像を表示し、判定が完了するたびに次の静止画をMacへ送る。フロントカメラを優先し、音声は取得しない。カメラ画像は長辺1,024 px以下のJPEGに切り出し、録画や画像保存はしない。結果に対応する直近の静止画も表示する。「停止」、ページを離れる操作、ページが非表示になる操作でカメラを止める。停止後に届いた処理中の結果は反映しない。通信や推論に失敗した場合も停止し、手動で再開できる。許可待ち中も「停止」で開始を取り消せる。

遠隔の端末でカメラを使う場合はHTTPSが必要。サーバをlocalhostのまま起動し、未使用のHTTPSポートを選んでTailscale Serveで転送する（同じポートの既存設定は上書きしない）。例えばアプリとHTTPSのポートがともに8017の場合：

```bash
tailscale serve --bg --https=8017 http://127.0.0.1:8017
```

Serveが出力するHTTPS URLを同じtailnet内の端末で開く。アプリ独自の認証はないため、アクセス制御はtailnet側に依存する。動作確認は模擬カメラ映像からの実モデル推論で行い、iPhone実機のカメラ権限、Safariの映像再生、表情変化への応答は別途確認する。

## API

- `GET /`：公式JavaScript SDK経由で動く判断デモ。入力文と質問JSONを編集できる。
- `GET /vendor/typesafe-sdk.mjs`：同一サーバから配信する公式SDK v0.6.0のESM配布物（MITライセンスは `src/systemone_workbench/vendor/LICENSE.typesafe-sdk`）。
- `POST /v1/systemone`：TypeSafeの`state`、`model`、`questions`形式。`choice`、`score`、`noul`を扱う。
- `GET /v1/models`：公式SDKで読めるモデル一覧。`jev-latest`は互換エイリアスであり、実体は`laya-multilingual-mlx`。ほかに`laya-multilingual-coreml`と`laya-multilingual-coreml-ane`を公開する。設定すれば、微調整したモデルと、Jeff・Kevなど別サーバへの転送も並ぶ（下記）。
- `GET /healthz`：モデルの読み込みが完了してから`ready`を返す。
- `GET /docs`：FastAPIの対話的なAPI仕様。

公式Python SDKからは`TypeSafeClient(api_key="local-demo", base_url="http://127.0.0.1:8017")`を使える。ローカルサーバはAPIキーを検証しない。リクエストの`model`で推論先を選び、応答には実際に判定したモデル名を返す。Snakeの`POST /snake/api/sessions`も任意の`{"model":"laya-multilingual-coreml-ane"}`を受け付け、省略時は従来どおりMLXを使う。

MLXのモデルは最大8,192トークン、汎用Core MLは最大1,024トークン、Neural Engine用の`laya-multilingual-coreml-ane`は最大96トークン。上限にはstateだけでなく質問、選択肢、特殊トークンも含む。MLXは既存の重みを使い、実行時の`max_len`だけを拡張する。短文を上限まで埋めて計算することはないが、実際の入力が長くなると推論時間とメモリ使用量は増える。多言語モデルの学習時の長さは1,024トークンであり、8,192トークン全域での判断精度を保証するものではない（[本家の長文対応と検証結果](https://github.com/NandhaKishorM/laya/blob/23a17522aa4942da6cce53a995a275760320b691/README.md)）。

state全体が質問の接頭辞とともに収まらないときは黙って切り詰めず422を返す。モデル間の暗黙の切り替えはしない。ただし、Laya内部の質問文や選択肢の圧縮まで防ぐものではない。Jev本体とは重み、精度、速度、コンテキスト長、確率の校正が異なる。互換性はこのAPIの形と公式SDKからの基本的な疎通を指し、Jevと同じ判断結果を意味しない。

## モデルのおすすめAPI（Jevにない拡張）

やりたいことを文章で送ると、どのモデルを使えばよいかを返す。Jevにも互換仕様にもない、このサーバ独自のAPIで、`/v1/systemone`とは別に動く。Jeff、Kev、Jev本体のように、このサーバでは提供していないモデルも候補に含め、起動方法も返す。

```bash
curl -s localhost:8017/v1/recommend -H 'content-type: application/json' -d '{
  "text": "配信中のAIキャラが、コメント全部に反応しすぎる。返事すべきかをリアルタイムで判定したい。データは外に出したくない"
}'
```

- `POST /v1/recommend`：`text`（必須、4,000文字まで）に、任意で`task`（用途を指定すると判定を省く）、`constraints`（`local_only`、`low_latency`、`long_text`、`negation`、`many_labels`。文章から読み取った値を上書きする）、`available_only`（このサーバで今使えるモデルだけに絞る）、`limit`（1〜7、既定3）を渡す。応答は、判定した用途と確信度、読み取った制約とその出どころ、順位づけした推薦（スコア、理由、警告、補足、使い方）、除外したモデルとその理由。
- `GET /v1/recommend/catalog`：用途の一覧と、モデルごとの実測値（遅延、選択肢の上限、否定形・長文の精度、用途別の適合度）。

用途は10種類（ターン制御、記憶判定、話題逸脱の検知、少数分類、多数分類、有害・スパム検知、段階評価、時系列の状態判定、文書間の関連判定、その他）。「その他」は生成・要約・推論のような選択肢から選ぶ判断ではない依頼で、モデルは推薦せず、生成モデルを使うよう返す。ローカル限定なら本文を外部へ送るJev本体を除外し、選択肢が上限を超えるならJeffを除外する。

用途の判定は、日本語の依頼文で学習した小さな文字n-gram分類器（`intent_model.json`、約280KB、純Pythonで0.1ms未満）で行い、判断モデルは呼ばない。ゼロショットのLayaに同じ判定をさせたところ57%程度にしか届かなかったため、この形にした。未使用の評価データ120件での用途の正答率は約90%（確信度0.5以上に限ると98%）。制約の読み取りは取りこぼしがあり、特に否定形と低遅延は見逃しやすいので、重要な条件は`constraints`で明示する。順位づけに使う数値は、日本語の自作データ・少数サンプル・1台のMacでの実測で、目安として使う（[調査メモ](docs/research/jev-vs-laya-case-studies.md)の第10節）。

## Jeff・Kevなど別のJev互換サーバをつなぐ

別のプロセスで動かしているJev互換サーバ（[Jeff](https://github.com/firelex/jeff)、[Kev](https://github.com/jaredpalmer/kev)など）を、環境変数 `SYSTEMONE_PROXY_MODELS` で登録すると、リクエストの`model`名でこのサーバから使える。書式は `名前=URL[|上流のモデル名]` をカンマ区切りにしたもの。

```bash
# 別のターミナルでそれぞれ起動しておく
python -m kev.serve --run jaredpalmer/kev-4b --port 8009                       # Kev（jaredpalmer/kev）
JEFF_BACKEND=mlx JEFF_CHECKPOINT=<Jeffのチェックポイント> PORT=8765 jeff-serve  # Jeff（firelex/jeff）

SYSTEMONE_PROXY_MODELS="kev-4b=http://127.0.0.1:8009,jeff-2b=http://127.0.0.1:8765" uv run systemone
```

上流のモデル名を省くと、名前の先頭部分から `kev-latest`、`jeff-latest` を使う（それ以外は `jev-latest`）。名前は `/v1/recommend` のカタログと同じ `kev-4b`、`kev-0.8b`、`jeff-2b`、`jeff-0.8b` にすると、おすすめAPIが利用可能なモデルとして扱い、`available_only`でも残る。

このサーバは`/v1/systemone`を上流へ転送し、応答の`model`を登録名に置き換えるだけで、重みも推論も上流のもの。確率は直接呼んだときと一致し、転送による遅延の増加は約1msだった。`/v1/models`には`backend: proxy`と上流のホスト・モデル名（`proxy:127.0.0.1:8009/kev-latest`）を出す。上流が止まっている、または遅いときは、別のモデルへ黙って切り替えず、接続できなければ502、混雑・時間切れなら503を返す。上流がリクエストを拒否したとき（Jeffの選択肢26個の上限など）は、その内容を422で返す。起動時に上流が応答しなくても起動は失敗せず、警告だけを出す（後から起動してよい）。Snakeデモには使わない。転送先はこの環境変数で設定した運用者の指定先だけで、リクエストからは指定できない。

## 微調整したモデルを載せる

Layaを自前データで微調整したチェックポイントは、環境変数 `SYSTEMONE_MLX_MODELS` で追加のモデルとして公開できる。書式は `名前=チェックポイント` をカンマ区切りにしたもので、チェックポイントはローカルのディレクトリかHugging Faceのid。

```bash
# 1. 微調整（eval/finetune/、PyTorchが必要）→ 出力ディレクトリ ft_out
# 2. MLX形式へ変換（既存の出力先には書き込まない）
uv run laya-mlx convert --model ft_out --output ~/models/laya-ja-turn-mlx
# 3. 追加モデルとして起動
SYSTEMONE_MLX_MODELS="laya-ja-turn-mlx=$HOME/models/laya-ja-turn-mlx" uv run systemone
```

リクエストの`model`に `laya-ja-turn-mlx` を指定すると、そのモデルで判定する。`/v1/models`にも並び、`jev-latest`は従来どおり`laya-multilingual-mlx`のまま変わらない。名前が既存のモデルや`jev-latest`と重なる場合、書式が不正な場合、チェックポイントを読み込めない場合は、無視せず起動に失敗する。ローカルの絶対パスは`/v1/models`に出さず`local:<ディレクトリ名>`と表示する。Snakeデモには追加モデルを使わない。

追加モデルは学習時の長さ（微調整の既定では1,024トークン）のままで動かし、内蔵の多言語モデルのように8,192トークンへ拡張しない。学習していない長さの精度は保証できないため。同梱の評価では、`laya-ja-turn-mlx`（日本語の会話ターン処理向けに微調整）を`model`に指定して、PyTorchで測った結果と同じ精度と、単発約7.5msの遅延を確認している（[調査メモ](docs/research/jev-vs-laya-case-studies.md)の第8節）。重みはこのリポジトリに含めない。

`laya-mlx`は校正温度を0.5〜5に切り詰めるため、微調整で得た温度がこの範囲を超えると、その分だけ確率の校正がずれる（警告が出る）。微調整したモデルは、学習した言い回しの質問（肯定形）だけで使う。否定形や反転した質問への頑健さは学習されていない。

## 動作確認

動作確認は、TypeSafe公式クライアントからローカルサーバへ接続して行う。

単体テストだけで完了とはせず、少なくとも次を確認する。

- TypeSafe公式クライアントからリクエストできる
- `laya-mlx` の判断結果をクライアントが受け取れる
- System One互換のエラー応答を確認できる
- `/v1/models` と `/healthz` が期待どおり動作する

## 参考実装・資料

- [`docs/research/jev-vs-laya-case-studies.md`](docs/research/jev-vs-laya-case-studies.md) — JevとLayaのケーススタディ調査と、日本語のローカル実測(`eval/ja/`)。

### Jev互換サーバ

- [`laya-serve`](https://pypi.org/project/laya-serve/) — Layaの重みをローカルで読み込み、`POST /v1/systemone`、`GET /v1/models`、`GET /healthz` を提供する先行実装。エンドポイント構成、モデル名の扱い、入力検証、エラー応答、preloadやAPIキーなどのサーバ運用面を参考にする。
- [`jev-compatible-server`](https://github.com/Hanno-Labs/jev-compatible-server) — 困ったときに確認する参考実装。複数の推論バックエンドを共通のtyped decision APIへ適合する設計、`choice`・`score`・`noul` の部分対応、対応不能な質問を明示する考え方を参考にする。ただし、このリポジトリの実装土台や対応backendの正本にはしない。

### Layaの各バックエンド

- [`laya-mlx`](https://github.com/mizorewww/laya-mlx) — Apple Silicon上でMLXを使ってLayaを実行する実装。Python API、モデルロード、言語ルーティング、入力長やバッチ処理の制約を確認し、最初のadapterの実装対象にする。
- [`laya-coreml`](https://github.com/mizorewww/laya-coreml) — Core ML / Neural Engine向けのLaya実装。MLXに依存しない実行経路、モデルの初期化、Core ML特有の入力・出力制約を確認し、2番目のadapterの参考にする。
- [`laya-onnx`](https://github.com/gqgs/laya-onnx) — ONNX Runtime / WebGPUを含むLaya実装。ONNXモデルの入出力と、ブラウザや別ランタイムへ展開する際の制約を確認し、3番目のadapterの参考にする。
- [`laya-multilingual-onnx`](https://huggingface.co/mizchi/laya-multilingual-onnx) — ONNX形式の公開モデル。モデルファイルをサーバ起動時にどう指定・検証するかを考えるときの参考にする。

### 公式契約

- [TypeSafe公式ドキュメント](https://docs.typesafe.ai/llms.txt) — System Oneの概念、API契約、SDK、primitive、モデル、制限の正本。実装時は必ず現行ドキュメントを読み、互換性をここから確認する。
- [TypeSafe公式Agent Skill](https://github.com/typesafe-ai/skills) — TypeSafeを使うワークフロー、公式ドキュメントの読み方、`choice`・`score`・`noul` の設計方針を確認するためのSkill。

参考実装をそのままコピーするのではなく、**TypeSafe公式の互換性を正本**とし、各実装からはサーバ構造・adapter境界・運用上の知見だけを取り入れる。

## ライセンス

本リポジトリのコードは [MIT License](LICENSE) で公開する。同梱するTypeSafe公式JavaScript SDKには、別途 [TypeSafeのMITライセンス](src/systemone_workbench/vendor/LICENSE.typesafe-sdk) が適用される。

## 開発状況

`laya-mlx` と `laya-coreml` のHTTP APIとWebデモを実装。`laya-onnx` は未対応。従来のプレビューはユーザーがiPhone実機で確認済み。公式SDKを使用する画面への変更後は、改めて実機で確認する。

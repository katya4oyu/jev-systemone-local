# jev-systemone-local

ローカルで動作する、Jev互換のSystem One APIサーバ。

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

Macだけで試すなら `uv run jev-systemone-local` で起動し、`http://127.0.0.1:8017/` を開く。iPhoneからTailscale経由で試すときはMacのTailscale IPv4アドレスだけで待ち受ける：

```bash
tailscale ip -4
uv run jev-systemone-local --host <MacのTailscale IPv4> --port 8017
```

iPhoneも同じtailnetへ接続し、`http://<MacのTailscale IPv4>:8017/` をSafariで開く。画面は同一サーバから配信するTypeSafe公式JavaScript SDK v0.6.0を使い、`models.list()` と `systemOne()` でMLXまたはCore MLの判定を呼び出す。ブラウザ用にはローカルサーバが検証しないダミーキーだけを指定する。実際のAPIキーをブラウザへ渡さない。同じサーバの `/snake` は独自のゲームAPIを使うため、SDK互換性の確認対象ではない。サーバにアプリ独自の認証はないため、公開ネットワークへはバインドしない。ブラウザとAPIの通信はHTTPであり、秘匿すべき文章を入力しない。

## Snakeデモ

`/snake` ではモデル、「開始」「一時停止」「新しいゲーム」、速度の上限を操作できる。盤面、方向ごとのモデル確率、モデルの第一候補、実行した方向、安全補助の介入、推論時間を表示する。各手の判定にはゲーム開始時に選んだモデルを使う。ゲーム盤面は `laya-mlx` 同梱の `SnakeGame`、各モデルの推論には対応する `LayaPolicy` を利用する。文章判定APIとSnakeでモデルを共有し、重みを二重にロードしない。

安全補助は元デモの既定どおり有効。元デモと同様にモデルは安全性や餌への進路の説明を入力として受けるので、盤面だけから戦略を学習した例ではない。画面に表示する確率はモデルの実出力で、移動後の盤面に添える「直前の判断」として扱う。推定値は校正された死亡確率ではない。ゲームはサーバのメモリに最大16件保持し、1時間操作がなければ期限切れになる。サーバ再起動でも失われる。速度の数値は上限であり、推論や通信が遅ければそれ以下になる。

## API

- `GET /`：公式JavaScript SDK経由で動く判断デモ。入力文と質問JSONを編集できる。
- `GET /vendor/typesafe-sdk.mjs`：同一サーバから配信する公式SDK v0.6.0のESM配布物（MITライセンスは `src/jev_systemone_local/vendor/LICENSE.typesafe-sdk`）。
- `POST /v1/systemone`：TypeSafeの`state`、`model`、`questions`形式。`choice`、`score`、`noul`を扱う。
- `GET /v1/models`：公式SDKで読めるモデル一覧。`jev-latest`は互換エイリアスであり、実体は`laya-multilingual-mlx`。ほかに`laya-multilingual-coreml`と`laya-multilingual-coreml-ane`を公開する。
- `GET /healthz`：モデルの読み込みが完了してから`ready`を返す。
- `GET /docs`：FastAPIの対話的なAPI仕様。

公式Python SDKからは`TypeSafeClient(api_key="local-demo", base_url="http://127.0.0.1:8017")`を使える。ローカルサーバはAPIキーを検証しない。リクエストの`model`で推論先を選び、応答には実際に判定したモデル名を返す。Snakeの`POST /snake/api/sessions`も任意の`{"model":"laya-multilingual-coreml-ane"}`を受け付け、省略時は従来どおりMLXを使う。

MLXと汎用Core MLのモデルは最大1,024トークン、Neural Engine用の`laya-multilingual-coreml-ane`は最大96トークン。state全体が質問の接頭辞とともに収まらないときは黙って切り詰めず422を返す。モデル間の暗黙の切り替えはしない。ただし、Laya内部の質問文や選択肢の圧縮まで防ぐものではない。Jev本体とは重み、精度、速度、コンテキスト長、確率の校正が異なる。互換性はこのAPIの形と公式SDKからの基本的な疎通を指し、Jevと同じ判断結果を意味しない。

## 動作確認

動作確認は、TypeSafe公式クライアントからローカルサーバへ接続して行う。

単体テストだけで完了とはせず、少なくとも次を確認する。

- TypeSafe公式クライアントからリクエストできる
- `laya-mlx` の判断結果をクライアントが受け取れる
- System One互換のエラー応答を確認できる
- `/v1/models` と `/healthz` が期待どおり動作する

## 参考実装・資料

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

## 開発状況

`laya-mlx` と `laya-coreml` のHTTP APIとWebデモを実装。`laya-onnx` は未対応。従来のプレビューはユーザーがiPhone実機で確認済み。公式SDKを使用する画面への変更後は、改めて実機で確認する。

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

APIサーバの構造や互換性の扱いは、先行実装の `laya-server` を参考にする。ただし、対応バックエンドとAPI契約はこのリポジトリで明示的に管理する。

## API

予定している主なエンドポイント：

- `POST /v1/systemone`
- `GET /v1/models`
- `GET /healthz`

詳細なリクエスト・レスポンス契約は、実装前にSystem One互換性を確認して定義する。

## 動作確認

動作確認は、TypeSafe公式クライアントからローカルサーバへ接続して行う。

単体テストだけで完了とはせず、少なくとも次を確認する。

- TypeSafe公式クライアントからリクエストできる
- `laya-mlx` の判断結果をクライアントが受け取れる
- System One互換のエラー応答を確認できる
- `/v1/models` と `/healthz` が期待どおり動作する

## 開発状況

リポジトリの初期作成と方針の記録のみ。サーバ実装はこれから。

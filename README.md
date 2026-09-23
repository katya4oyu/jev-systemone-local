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

## 参考実装・資料

### Jev互換サーバ

- [`laya-serve`](https://pypi.org/project/laya-serve/) — Layaの重みをローカルで読み込み、`POST /v1/systemone`、`GET /v1/models`、`GET /healthz` を提供する先行実装。エンドポイント構成、モデル名の扱い、入力検証、エラー応答、preloadやAPIキーなどのサーバ運用面を参考にする。
- [`jev-compatible-server`](https://github.com/Hanno-Labs/jev-compatible-server) — オープンな意思決定モデルをJev互換APIで提供する別実装。複数の推論バックエンドを共通のtyped decision APIへ適合する設計、`choice`・`score`・`noul` の部分対応、対応不能な質問を明示する考え方を参考にする。

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

リポジトリの初期作成と方針の記録のみ。サーバ実装はこれから。

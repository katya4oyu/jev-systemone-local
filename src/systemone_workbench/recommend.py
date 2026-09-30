"""Model recommendation: describe what you want to do, get the System One model that fits.

Not part of the Jev API. The task type is classified by a small character n-gram model trained on Japanese
example requests (intent_classifier.py; the served Laya was tried first and reached only ~57%, see
eval/recommend); constraints (local only, low latency, label count, long text, negated questions) are read from
the text with patterns or passed explicitly. Models are then ranked with a capability catalog built from the
measurements in docs/research/jev-vs-laya-case-studies.md.

The catalog covers models this server can serve and models it cannot (Jev API, Jeff, Kev); every entry says which.
Numbers are small-sample measurements on one Mac (M4 Pro) with self-written Japanese data. Treat them as a guide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .intent_classifier import load_classifier

# --- task taxonomy -------------------------------------------------------------------------------------------

TASKS: dict[str, dict[str, str]] = {
    "turn_taking": {"label": "ターン制御",
                    "desc": "音声・チャットのAIが今応答すべきか相槌だけにするかを決める、またはユーザーの発話が終わったかを判定する"},
    "memory": {"label": "記憶判定", "desc": "会話中の発言や事実を、長期記憶に保存すべきか判断する"},
    "topic_shift": {"label": "話題逸脱の検知", "desc": "会話の話題が逸れた、または別の話題に分岐したことを検出する"},
    "classify": {"label": "少数分類", "desc": "問い合わせ・チケット・文書・メールを、少数（20以下）のカテゴリに分類・振り分けする"},
    "classify_many": {"label": "多数分類", "desc": "数十から数百の非常に多くのカテゴリやラベルの中から選ぶ分類"},
    "moderation": {"label": "有害・スパム検知", "desc": "有害・攻撃的・スパム・プロンプトインジェクション・個人情報などを、はい／いいえで検知する"},
    "rating": {"label": "段階評価", "desc": "感情の強さ・緊急度・品質・満足度などを段階やスコアで評価する"},
    "timeseries": {"label": "時系列の状態判定", "desc": "センサー・ログ・指標などの時系列について、傾向・異常・スパイク・警告レベルを判定する"},
    "relevance": {"label": "文書間の関連判定", "desc": "文書・ページ・ノートの間のリンク候補、重複、同じ概念か、分割すべき箇所を判定する"},
    "other": {"label": "その他（生成・推論など）",
              "desc": "文章生成、要約、翻訳、コード作成、複数手順の推論など、選択肢から選ぶ判断ではないもの"},
}

# --- model catalog -------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    id: str
    label: str
    runtime: str                    # "local" (served by this project) or "external" (separate server / hosted API)
    p50_ms: float                   # single-question latency (client-measured; Jev includes the network)
    local: bool                     # request text stays on this machine
    max_options: int | None         # hard cap on choice options, None = none known
    negation: float                 # accuracy on negated yes/no questions (0-1)
    long_text: float                # accuracy with ~8k-token inputs (0-1)
    fit: dict[str, float | None]    # per-task suitability 0-1 from measurements; None = not measured
    notes: tuple[str, ...] = ()
    how: str = ""


CATALOG: tuple[Profile, ...] = (
    Profile(
        "laya-ja-turn-mlx", "Laya（日本語ターン処理で微調整）", "local", 7.5, True, None, 0.25, 0.2,
        {"turn_taking": 0.95, "memory": 0.98, "topic_shift": 0.97, "classify": 0.75, "classify_many": 0.2,
         "moderation": 0.9, "rating": None, "timeseries": None, "relevance": None},
        ("会話ターン処理向けに微調整済み。学習した言い回し（肯定形）の質問だけで使う", "学習時の長さ1,024トークンまで",
         "多ラベル分類は微調整前より劣化（20ラベルで0.79→0.71）"),
        "SYSTEMONE_MLX_MODELS に登録して model に指定（README「微調整したモデルを載せる」）"),
    Profile(
        "laya-multilingual-mlx", "Laya multilingual（微調整なし）", "local", 7.2, True, None, 0.35, 0.5,
        {"turn_taking": 0.3, "memory": 0.58, "topic_shift": 0.69, "classify": 0.81, "classify_many": 0.33,
         "moderation": 0.7, "rating": 0.96, "timeseries": 0.6, "relevance": 0.45},
        ("日本語のゼロショットは語用論的な判断が弱く、信頼度が過信気味", "時系列は数値を日本語の要約にしてから渡すと使える"),
        "このサーバの model に laya-multilingual-mlx（または jev-latest）"),
    Profile(
        "kev-4b", "Kev 4B", "external", 47, True, None, 1.0, 1.0,
        {"turn_taking": 0.72, "memory": 0.99, "topic_shift": 1.0, "classify": 1.0, "classify_many": 1.0,
         "moderation": 1.0, "rating": 0.98, "timeseries": 0.98, "relevance": 0.93},
        ("ローカルの汎用判断で最もJevに近い", "Mac(MLX)で単発47ms・5問149ms。要32GBクラスのメモリ"),
        "別プロセスで uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009（jaredpalmer/kev）を起動し、SYSTEMONE_PROXY_MODELS=kev-4b=http://127.0.0.1:8009 で登録"),
    Profile(
        "kev-0.8b", "Kev 0.8B", "external", 11, True, None, 0.75, 0.92,
        {"turn_taking": 0.55, "memory": 0.8, "topic_shift": 0.96, "classify": 0.96, "classify_many": 0.75,
         "moderation": 0.9, "rating": 0.71, "timeseries": 0.8, "relevance": 0.82},
        ("小型で速い（単発11ms）が、score型と時系列は弱め",),
        "別プロセスで python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009 を起動し、SYSTEMONE_PROXY_MODELS=kev-0.8b=http://127.0.0.1:8009 で登録"),
    Profile(
        "jeff-2b", "Jeff 2B", "external", 62, True, 26, 0.7, 1.0,
        {"turn_taking": 0.58, "memory": 0.96, "topic_shift": 0.96, "classify": 0.96, "classify_many": 0.0,
         "moderation": 0.85, "rating": 0.96, "timeseries": 0.95, "relevance": 0.9},
        ("選択肢は最大26個", "READMEは英語のみだが日本語でも実用的だった"),
        "別プロセスで JEFF_BACKEND=mlx JEFF_CHECKPOINT=<Jeff-Qwen3.5-2B> PORT=8765 jeff-serve（firelex/jeff、uv 0.12.19以上）を起動し、SYSTEMONE_PROXY_MODELS=jeff-2b=http://127.0.0.1:8765 で登録"),
    Profile(
        "jeff-0.8b", "Jeff 0.8B", "external", 27, True, 26, 0.75, 0.92,
        {"turn_taking": 0.48, "memory": 0.93, "topic_shift": 0.9, "classify": 0.96, "classify_many": 0.0,
         "moderation": 0.85, "rating": 0.88, "timeseries": 0.75, "relevance": 0.88},
        ("選択肢は最大26個", "小型で速く、少数分類・関連判定は実用的"),
        "別プロセスで JEFF_BACKEND=mlx JEFF_CHECKPOINT=<Jeff-Qwen3.5-0.8B> PORT=8765 jeff-serve を起動し、SYSTEMONE_PROXY_MODELS=jeff-0.8b=http://127.0.0.1:8765 で登録"),
    Profile(
        "jev", "Jev（ホスト型API）", "external", 240, False, None, 1.0, 1.0,
        {"turn_taking": 0.85, "memory": 1.0, "topic_shift": 1.0, "classify": 0.92, "classify_many": 0.88,
         "moderation": 1.0, "rating": 0.99, "timeseries": 0.98, "relevance": 0.95},
        ("最も安定して高精度", "リクエストの本文がTypeSafeのAPIへ送られる", "遅延は約240ms（ネットワーク込）"),
        "TypeSafeClient(api_key=$TYPESAFE_API_KEY)（typesafe-sdk）"),
)
BY_ID = {p.id: p for p in CATALOG}
DEFAULT_FIT = 0.35  # used when a task was never measured for a model, always with a warning

# --- constraint detection ------------------------------------------------------------------------------------

RE_LOCAL = re.compile(
    r"ローカル|オフライン|オンプレ|オンデバイス|エッジ|手元|閉域|端末(?:の)?(?:内|外)|外には出|外に出|"
    r"外部(?:に|へ|API)?.{0,4}(?:送|出|漏|使え|使わ)|社外(?:に|へ)|機密|プライバシ|"
    r"クラウド(?:に|へ|には).{0,4}(?:送|上げ)|ネット(?:に|へ)?(?:は)?つながっ?て?い?な|インターネット(?:に|へ)?(?:は)?(?:接続|繋)")
RE_LATENCY = re.compile(
    r"リアルタイム|低遅延|低レイテンシ|即座|瞬時|ミリ秒|\d+\s*ms|同時通訳|ストリーミング|遅いと|待たせ|体験が悪|タイムラグ|"
    r"送信前|その場で|割り込み|間髪")
RE_LONG = re.compile(
    r"長文|長大|議事録|明細書|数千|数万|万字|(?:\d[\d,]{3,})\s*(?:文字|字|トークン|単語|words?)|PDF|数ページ|"
    r"(?:かなり|とても|非常に|すごく|結構)長|長すぎ|分量|長い(?:文書|ドキュメント|テキスト|ページ)|文字起こし")
RE_NEG = re.compile(r"でない|ではない|じゃない|以外|除外|外す|外した|含まない|なくていい|しない.{0,6}(?:検出|判定|抽出)")
RE_COUNT = re.compile(r"(\d[\d,]*)\s*(?:種類|クラス|ラベル|カテゴリー?|択|個以上|個の(?:分類|カテゴリ|ラベル|インテント)|つの(?:カテゴリ|分類)|つに|通りに)")
RE_QUOTED = re.compile(r"「[^」]{1,12}」")
RE_MANY_WORDS = re.compile(r"数百(?:種類|クラス|カテゴリ|ラベル|の)|数十(?:種類|クラス|カテゴリ|ラベル|の)|多数の(?:カテゴリ|ラベル|クラス)|膨大な(?:カテゴリ|ラベル)")

def detect_constraints(text: str) -> dict[str, Any]:
    found: dict[str, Any] = {}
    if RE_LOCAL.search(text):
        found["local_only"] = True
    if RE_LATENCY.search(text):
        found["low_latency"] = True
    if RE_LONG.search(text):
        found["long_text"] = True
    if RE_NEG.search(text):
        found["negation"] = True
    counts = [int(m.group(1).replace(",", "")) for m in RE_COUNT.finditer(text)]
    quoted = len(RE_QUOTED.findall(text))
    if counts:
        found["many_labels"] = max(counts)
    elif quoted >= 3:  # labels written inline, e.g. 「請求」「技術」「解約」
        found["many_labels"] = quoted
    elif RE_MANY_WORDS.search(text):
        found["many_labels"] = 100
    return found


# --- task detection ------------------------------------------------------------------------------------------

def detect_task(text: str) -> dict[str, Any]:
    probs = load_classifier().predict_proba(text)
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return {"task": ranked[0][0], "confidence": round(ranked[0][1], 4),
            "candidates": [{"task": k, "label": TASKS[k]["label"], "probability": round(p, 4)} for k, p in ranked[:3]]}


# --- ranking -------------------------------------------------------------------------------------------------

LOW_LATENCY_MS = 50.0
SMALL_LABEL_LIMIT = 20  # up to this many labels counts as 少数分類


def rank_models(task: str, constraints: dict[str, Any], served: set[str], *, available_only: bool = False,
                limit: int = 3) -> tuple[list[dict], list[dict]]:
    ranked: list[dict] = []
    excluded: list[dict] = []
    n_labels = constraints.get("many_labels")
    for p in CATALOG:
        reasons: list[str] = []
        warnings: list[str] = []
        if available_only and p.id not in served:
            excluded.append({"model": p.id, "reason": "このサーバでは提供していない（available_only）"})
            continue
        if constraints.get("local_only") and not p.local:
            excluded.append({"model": p.id, "reason": "本文が外部APIへ送られるため、ローカル限定の条件に合わない"})
            continue
        if n_labels and p.max_options and n_labels > p.max_options:
            excluded.append({"model": p.id, "reason": f"選択肢の上限が{p.max_options}個で、{n_labels}ラベルは扱えない"})
            continue
        fit = p.fit.get(task)
        if fit is None:
            fit = DEFAULT_FIT
            warnings.append("このタスクは未評価（暫定スコア）")
        else:
            reasons.append(f"「{TASKS[task]['label']}」の実測での適合度 {fit:.2f}")
        score = fit
        if constraints.get("low_latency"):
            if p.p50_ms > LOW_LATENCY_MS:
                score *= 0.6
                warnings.append(f"低遅延の条件に対して単発{p.p50_ms:.0f}msは遅め")
            else:
                reasons.append(f"単発{p.p50_ms:.0f}msで低遅延")
        if n_labels and n_labels > 20:
            factor = {"laya": 0.4}.get(p.id.split("-")[0], 1.0)
            if factor < 1:
                warnings.append(f"{n_labels}ラベルは苦手（20ラベル超で精度が急落）")
            score *= factor
        if constraints.get("negation") and p.negation < 0.9:
            score *= 0.5 + 0.5 * p.negation
            warnings.append(f"否定形の質問は不得意（正答率{p.negation:.2f}）。肯定形に書き換える")
        if constraints.get("long_text") and p.long_text < 0.9:
            score *= 0.5 + 0.5 * p.long_text
            warnings.append(f"長文は苦手（約8kトークンで{p.long_text:.2f}）。要点を先頭に置く・分割する")
        if constraints.get("local_only") and p.local:
            reasons.append("本文がこのマシンの外へ出ない")
        ranked.append({"model": p.id, "label": p.label, "score": round(score, 3), "served_here": p.id in served,
                       "latency_ms_p50": p.p50_ms, "reasons": reasons, "warnings": warnings, "notes": list(p.notes),
                       "how_to_use": "model に " + p.id + " を指定" if p.id in served else p.how})
    ranked.sort(key=lambda r: (r["score"], r["served_here"]), reverse=True)
    for i, r in enumerate(ranked, 1):
        r["rank"] = i
    return ranked[:limit], excluded


def recommend(text: str, served: set[str], *, task: str | None = None, constraints: dict[str, Any] | None = None, available_only: bool = False,
              limit: int = 3) -> dict[str, Any]:
    if task is not None and task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {sorted(TASKS)}")
    detected = None
    if task is None:
        detected = detect_task(text)
        task = detected["task"]
    merged = detect_constraints(text)
    sources = {k: "text" for k in merged}
    for key, value in (constraints or {}).items():
        merged[key] = value
        sources[key] = "request"
    adjusted = None
    n_labels = merged.get("many_labels")
    if detected and n_labels and task in ("classify", "classify_many"):
        # A stated label count is more reliable than the text classifier for telling the two apart.
        scaled = "classify_many" if n_labels > SMALL_LABEL_LIMIT else "classify"
        if scaled != task:
            adjusted, task = task, scaled
    result: dict[str, Any] = {
        "task": {"id": task, "label": TASKS[task]["label"], "source": "request" if detected is None else "detected",
                 **({"confidence": detected["confidence"], "candidates": detected["candidates"]} if detected else {})},
        "constraints": {"values": merged, "source": sources},
        "recommendations": [], "excluded": [], "notes": [],
    }
    if adjusted:
        result["notes"].append(f"ラベル数（{n_labels}）から「{TASKS[adjusted]['label']}」を「{TASKS[task]['label']}」に修正した")
    if task == "other":
        result["notes"].append("選択肢から選ぶ判断（System One）向きではない依頼に見える。生成・要約・推論は生成モデル（LLM）を使う")
        return result
    if detected and detected["confidence"] < 0.5:
        result["notes"].append("タスクの判定に自信がない。task を明示すると確実")
    result["recommendations"], result["excluded"] = rank_models(task, merged, served, available_only=available_only, limit=limit)
    result["notes"].append("数値は日本語の自作データ・少数サンプル・1台のMacでの実測。目安として使い、自分のデータで確かめる")
    return result


def catalog_view(served: set[str]) -> dict[str, Any]:
    return {"tasks": [{"id": k, **v} for k, v in TASKS.items()],
            "models": [{"id": p.id, "label": p.label, "served_here": p.id in served, "local": p.local,
                        "latency_ms_p50": p.p50_ms, "max_options": p.max_options, "negation_accuracy": p.negation,
                        "long_text_accuracy": p.long_text, "fit": p.fit, "notes": list(p.notes), "how_to_use": p.how}
                       for p in CATALOG]}

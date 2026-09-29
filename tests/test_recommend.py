import pytest
from fastapi.testclient import TestClient

from systemone_workbench.app import create_app
from systemone_workbench.intent_classifier import load_classifier
from systemone_workbench.recommend import CATALOG, TASKS, detect_constraints, detect_task, rank_models, recommend


def test_classifier_knows_every_task_and_returns_a_distribution():
    classifier = load_classifier()
    assert set(classifier.classes) == set(TASKS)
    probs = classifier.predict_proba("音声アバターの相槌のタイミングを判定したい")
    assert abs(sum(probs.values()) - 1) < 1e-9 and max(probs, key=probs.get) == "turn_taking"
    assert abs(sum(classifier.predict_proba("").values()) - 1) < 1e-9  # empty text falls back to the intercept


@pytest.mark.parametrize("text,task", [
    ("問い合わせメールを担当部署ごとに振り分けたい", "classify"),
    ("チャットの話題が逸れたのを検出したい", "topic_shift"),
    ("センサー値の異常を判定したい", "timeseries"),
    ("ブログ記事を自動で書いてほしい", "other"),
])
def test_detects_obvious_requests(text, task):
    assert detect_task(text)["task"] == task


@pytest.mark.parametrize("text,expected", [
    ("外部APIに送れない社内データを分類したい", {"local_only": True}),
    ("音声アバターの相槌をリアルタイムに判定したい", {"low_latency": True}),
    ("120種類のカテゴリに分類したい", {"many_labels": 120}),
    ("数百カテゴリの商品分類", {"many_labels": 100}),
    ("議事録の長文から判定したい", {"long_text": True}),
    ("攻撃的でない投稿だけを抽出したい", {"negation": True}),
    ("問い合わせを分類したい", {}),
])
def test_detect_constraints(text, expected):
    assert detect_constraints(text) == expected


def test_local_only_excludes_hosted_jev():
    ranked, excluded = rank_models("classify", {"local_only": True}, set())
    assert "jev" not in [r["model"] for r in ranked]
    assert [e["model"] for e in excluded] == ["jev"]


def test_label_cap_excludes_jeff_and_prefers_kev_4b_for_many_labels():
    ranked, excluded = rank_models("classify_many", {"many_labels": 80}, set(), limit=7)
    assert {e["model"] for e in excluded} == {"jeff-2b", "jeff-0.8b"}
    assert ranked[0]["model"] == "kev-4b"
    laya = next(r for r in ranked if r["model"] == "laya-multilingual-mlx")
    assert any("20ラベル超" in w for w in laya["warnings"])
    assert laya["notes"]  # informational notes are kept apart from warnings


def test_turn_taking_prefers_fine_tuned_laya_and_marks_served():
    ranked, _ = rank_models("turn_taking", {"low_latency": True}, {"laya-ja-turn-mlx", "laya-multilingual-mlx"})
    assert ranked[0]["model"] == "laya-ja-turn-mlx" and ranked[0]["served_here"]
    assert ranked[0]["how_to_use"] == "model に laya-ja-turn-mlx を指定"
    jev = next(r for r in rank_models("turn_taking", {"low_latency": True}, set(), limit=7)[0] if r["model"] == "jev")
    assert any("240ms" in w for w in jev["warnings"])


def test_negation_and_unmeasured_task_are_flagged():
    ranked, _ = rank_models("moderation", {"negation": True}, set(), limit=7)
    laya = next(r for r in ranked if r["model"] == "laya-ja-turn-mlx")
    assert any("否定形" in w for w in laya["warnings"])
    rated, _ = rank_models("rating", {}, set(), limit=7)
    assert any("未評価" in w for w in next(r for r in rated if r["model"] == "laya-ja-turn-mlx")["warnings"])


def test_available_only_limits_to_served_models():
    ranked, _ = rank_models("classify", {}, {"laya-multilingual-mlx"}, available_only=True, limit=7)
    assert [r["model"] for r in ranked] == ["laya-multilingual-mlx"]


def test_recommend_other_is_not_a_system_one_task():
    result = recommend("ブログ記事を書いてほしい", set(), task="other")
    assert result["task"]["id"] == "other" and result["recommendations"] == []
    assert any("生成モデル" in n for n in result["notes"])


def test_explicit_task_and_constraints_override_detection():
    result = recommend("何でもいい", set(), task="memory", constraints={"local_only": True})
    assert result["task"]["source"] == "request" and "confidence" not in result["task"]
    assert result["constraints"]["source"] == {"local_only": "request"}
    detected = recommend("音声アバターの応答判断をリアルタイムで", set())
    assert detected["task"]["source"] == "detected" and detected["constraints"]["source"] == {"low_latency": "text"}


def test_catalog_scores_stay_within_range():
    for p in CATALOG:
        assert all(v is None or 0 <= v <= 1 for v in p.fit.values()), p.id
        assert set(p.fit) <= set(TASKS), p.id


class FakeBackend:
    backend_name = "laya-mlx"
    model_name = "laya-multilingual-mlx"
    checkpoint = "test-only"

    def evaluate(self, state, questions):
        raise AssertionError("/v1/recommend must not call the decision models")


def test_http_recommend_and_catalog():
    with TestClient(create_app(FakeBackend())) as client:
        body = client.post("/v1/recommend", json={"text": "音声アバターの応答判断をリアルタイムで"}).json()
        assert body["task"]["id"] == "turn_taking" and body["task"]["source"] == "detected"
        assert body["constraints"]["values"]["low_latency"] is True
        assert body["recommendations"][0]["rank"] == 1
        catalog = client.get("/v1/recommend/catalog").json()
        assert {m["id"] for m in catalog["models"]} >= {"jev", "kev-4b", "jeff-2b", "laya-multilingual-mlx"}
        served = {m["id"] for m in catalog["models"] if m["served_here"]}
        assert served == {"laya-multilingual-mlx"}


@pytest.mark.parametrize("payload", [
    {"text": ""}, {"text": "   "}, {"text": "x", "task": "nope"}, {"text": "x", "constraints": {"bogus": True}},
    {"text": "x", "constraints": {"many_labels": 0}}, {"text": "x", "limit": 0}, {"text": "x" * 4001}, {"text": "x", "extra": 1},
])
def test_http_validation_is_explicit(payload):
    with TestClient(create_app(FakeBackend())) as client:
        assert client.post("/v1/recommend", json=payload).status_code == 422


def test_stated_label_count_decides_between_few_and_many_labels():
    many = recommend("問い合わせを120種類のカテゴリに分類したい", set())
    assert many["task"]["id"] == "classify_many" and many["constraints"]["values"]["many_labels"] == 120
    few = recommend("問い合わせを3つに振り分けたい。カテゴリ分類のイメージです", set())
    assert few["task"]["id"] == "classify" and few["constraints"]["values"]["many_labels"] == 3
    # with a large count the small-label-only models are excluded and the label cap is reported
    assert {e["model"] for e in many["excluded"]} == {"jeff-2b", "jeff-0.8b"}

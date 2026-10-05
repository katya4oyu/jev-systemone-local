from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.vision.jaffe import (
    build_inventory,
    classification_metrics,
    make_record,
    normalize_image_key,
    parse_primary_ratings,
    rating_metrics,
)


def test_primary_rating_parser_excludes_separate_fear_excluded_study_and_preserves_ids():
    text = """# HAP SAD SUR ANG DIS FEA PIC
1 4.00 2.00 1.00 1.00 1.00 1.00 AA-HA1
2 2.00 3.00 2.00 1.00 1.00 1.00 YN-HA2
Semantic Ratings Data (Fear Excluded)
1 4.00 2.00 1.00 1.00 1.00 AA-HA1
"""
    ratings = parse_primary_ratings(text)
    assert ratings == {
        "AA-HA1": (4.0, 2.0, 1.0, 1.0, 1.0, 1.0),
        "YN-HA2": (2.0, 3.0, 2.0, 1.0, 1.0, 1.0),
    }


def test_image_key_parses_official_filename_without_repairing_unfamiliar_tokens():
    assert normalize_image_key("YM.HA2.53.tiff") == "YM-HA2"
    assert normalize_image_key("YN.HA2.53.tiff") == "YN-HA2"
    assert normalize_image_key("not-a-jaffe-image.png") is None


def test_classification_metrics_count_failed_rows_in_total_accuracy_denominator():
    result = classification_metrics(
        golds=["neutral", "happy", "sad"],
        predictions=["neutral", None, "happy"],
        labels=["neutral", "happy", "sad"],
    )
    assert result["n_total"] == 3
    assert result["n_valid"] == 2
    assert result["errors"] == 1
    assert result["accuracy"] == 0.333333
    assert result["macro_f1"] == 0.333333
    assert result["confusion"]["happy"]["happy"] == 0


def test_rating_metrics_converts_zero_based_model_score_to_human_one_to_five_scale():
    result = rating_metrics(golds=[1.0, 5.0, 3.0], predictions=[0.0, 4.0, 2.0])
    assert result["n_total"] == 3
    assert result["n_valid"] == 3
    assert result["errors"] == 0
    assert result["mae"] == 0.0
    assert result["pearson_r"] == 1.0


def test_rating_correlation_is_null_when_one_series_is_constant():
    result = rating_metrics(golds=[2.0, 2.0, 2.0], predictions=[0.0, 2.0, 4.0])
    assert result["pearson_r"] is None
    assert result["mae"] == 1.666667


def test_inventory_preserves_unmatched_source_ids_without_aliasing_or_repair():
    rows = {
        "AA-HA1": (4.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "YN-HA2": (4.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    }
    inventory = build_inventory(
        [Path("AA.HA1.1.tiff"), Path("AA.NE1.2.tiff")],
        rows,
    )
    assert inventory["matched_rating_count"] == 1
    assert inventory["image_without_rating_ids"] == ["AA-NE1"]
    assert inventory["rating_without_image_ids"] == ["YN-HA2"]


def test_model_input_uses_one_fixed_state_and_seven_questions_without_image_metadata():
    request = make_record(None)
    assert "images" not in request
    assert len(request["questions"]) == 7
    assert request["questions"]["expression"]["type"] == "choice"
    assert sum(q["type"] == "score" for q in request["questions"].values()) == 6
    serialized = str(request)
    assert "filename" not in serialized.lower()
    assert "YN-HA2" not in serialized
    assert "JAFFE" not in serialized

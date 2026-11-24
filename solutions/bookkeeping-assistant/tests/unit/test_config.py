from decimal import ROUND_UP, Context, Decimal, Inexact, Rounded, localcontext
from typing import Any

import pytest
from pydantic import ValidationError

from bookkeeping_assistant.config import ReviewConfig, ScoreWeights


def test_defaults() -> None:
    config = ReviewConfig()

    assert config.missing_document_threshold == Decimal("75.00")
    assert (config.receipt_days_before, config.receipt_days_after) == (5, 2)
    assert (config.invoice_days_before, config.invoice_days_after) == (60, 2)
    assert config.amount_close_ratio == Decimal("0.05")
    assert config.tip_hint_ratio == Decimal("0.20")
    assert config.high_severity_delta == Decimal("50.00")
    assert config.high_severity_ratio == Decimal("0.10")
    assert config.arithmetic_tolerance == Decimal("0.01")
    assert config.vendor_min_similarity == 0.6
    assert config.ambiguity_margin == 0.05
    assert config.weights == ScoreWeights(amount_exact=0.5, amount_close=0.25, date=0.2, vendor=0.3)


def test_money_and_ratio_fields_are_decimal() -> None:
    config = ReviewConfig()

    for name in (
        "missing_document_threshold",
        "amount_close_ratio",
        "tip_hint_ratio",
        "high_severity_delta",
        "high_severity_ratio",
        "arithmetic_tolerance",
    ):
        assert isinstance(getattr(config, name), Decimal), name


def test_json_round_trip_serializes_decimals_as_strings() -> None:
    config = ReviewConfig()

    payload = config.model_dump(mode="json")

    assert payload["missing_document_threshold"] == "75.00"
    assert payload["amount_close_ratio"] == "0.05"
    assert ReviewConfig.model_validate_json(config.model_dump_json()) == config


def test_partial_override_from_json() -> None:
    config = ReviewConfig.model_validate_json(
        '{"missing_document_threshold": "0", "invoice_days_before": 30}'
    )

    assert config.missing_document_threshold == Decimal("0.00")
    assert config.invoice_days_before == 30
    assert config.receipt_days_before == 5  # untouched default


@pytest.mark.parametrize(
    "overrides",
    [
        {"missing_document_threshold": "-1.00"},
        {"missing_document_threshold": 75.0},  # float
        {"missing_document_threshold": "75.001"},
        {"amount_close_ratio": "1.5"},
        {"amount_close_ratio": "-0.01"},
        {"amount_close_ratio": 0.05},  # float
        {"receipt_days_before": -1},
        {"invoice_days_before": 400},
        {"vendor_min_similarity": 1.5},
        {"ambiguity_margin": -0.1},
        {"unknown_setting": 1},
    ],
)
def test_rejects_invalid_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReviewConfig.model_validate(overrides)


def test_config_is_frozen() -> None:
    config = ReviewConfig()

    with pytest.raises(ValidationError):
        config.receipt_days_before = 10  # type: ignore[misc]


@pytest.mark.parametrize(
    ("weights", "message"),
    [
        ({"amount_exact": 0.6}, "must equal 1.0"),
        ({"vendor": 0.0}, "must equal 1.0"),
        ({"amount_exact": 0.3, "amount_close": 0.4, "date": 0.3, "vendor": 0.4}, "must not exceed"),
    ],
)
def test_score_weights_validation(weights: dict[str, float], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ScoreWeights.model_validate(weights)


def test_score_weights_allow_rebalancing() -> None:
    weights = ScoreWeights(amount_exact=0.6, amount_close=0.3, date=0.1, vendor=0.3)

    assert weights.amount_exact == 0.6


@pytest.mark.parametrize(
    ("amount_exact", "date", "vendor"),
    [(0.33, 0.56, 0.11), (0.34, 0.55, 0.11), (0.7, 0.2, 0.1), (0.1, 0.2, 0.7)],
)
def test_score_weights_tolerate_float_rounding(
    amount_exact: float, date: float, vendor: float
) -> None:
    weights = ScoreWeights(amount_exact=amount_exact, amount_close=0.0, date=date, vendor=vendor)

    assert weights.vendor == vendor


@pytest.mark.parametrize(
    "overrides",
    [
        {"receipt_days_before": True},
        {"receipt_days_after": False},
        {"invoice_days_before": True},
        {"invoice_days_after": False},
        {"vendor_min_similarity": True},
        {"ambiguity_margin": False},
        {"missing_document_threshold": True},
        {"amount_close_ratio": True},
        {"weights": {"amount_exact": True, "amount_close": 0.0, "date": False, "vendor": False}},
        {"weights": {"amount_close": False}},
    ],
)
def test_rejects_boolean_numeric_values(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        ReviewConfig.model_validate(overrides)


@pytest.mark.parametrize(
    "json_text",
    [
        '{"receipt_days_before": true}',
        '{"vendor_min_similarity": false}',
        '{"weights": {"amount_exact": true, "amount_close": 0, "date": 0, "vendor": 0}}',
    ],
)
def test_rejects_boolean_numeric_values_from_json(json_text: str) -> None:
    with pytest.raises(ValidationError):
        ReviewConfig.model_validate_json(json_text)


def test_numeric_inputs_still_accepted() -> None:
    """Ints, floats, and numeric strings remain valid where they were before."""
    config = ReviewConfig.model_validate_json(
        '{"receipt_days_before": 7, "invoice_days_after": "3", "vendor_min_similarity": 1,'
        ' "ambiguity_margin": "0.1",'
        ' "weights": {"amount_exact": 1, "amount_close": 0.5, "date": 0, "vendor": 0}}'
    )

    assert config.receipt_days_before == 7
    assert config.invoice_days_after == 3
    assert config.vendor_min_similarity == 1.0
    assert config.ambiguity_margin == 0.1
    assert config.weights.amount_exact == 1.0


@pytest.mark.parametrize(
    "context",
    [Context(prec=1), Context(prec=2, rounding=ROUND_UP, traps=[Inexact, Rounded])],
)
def test_decimal_fields_ignore_caller_decimal_context(context: Context) -> None:
    with localcontext(context):
        config = ReviewConfig.model_validate(
            {"missing_document_threshold": "1234.56", "amount_close_ratio": "0.0525"}
        )

    assert str(config.missing_document_threshold) == "1234.56"
    assert config.amount_close_ratio == Decimal("0.0525")

from decimal import (
    ROUND_DOWN,
    ROUND_UP,
    Context,
    Decimal,
    Inexact,
    InvalidOperation,
    Rounded,
    getcontext,
    localcontext,
)

import pytest
from pydantic import BaseModel, ValidationError

from bookkeeping_assistant.money import (
    MAX_INTEGER_DIGITS,
    MONEY_CONTEXT,
    AmountParseError,
    Money,
    decimal_context_guard,
    parse_amount,
)

# ----------------------------------------------------------------------------- parse_amount


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("12", "12.00"),
        ("12.5", "12.50"),
        ("12.05", "12.05"),
        ("0", "0.00"),
        ("0.00", "0.00"),
        ("-0.00", "0.00"),  # negative zero collapses
        ("(0.00)", "0.00"),
        (".50", "0.50"),
        ("$.50", "0.50"),
        ("-12.00", "-12.00"),
        ("+12.00", "12.00"),
        ("$1,234.56", "1234.56"),
        ("1,234,567.89", "1234567.89"),
        ("1234567.89", "1234567.89"),
        ("(12.00)", "-12.00"),
        ("($1,234.56)", "-1234.56"),
        ("-$12.00", "-12.00"),
        ("$-12.00", "-12.00"),
        ("  $ 12.00  ", "12.00"),
        ("- 12.00", "-12.00"),
        ("( 12.00 )", "-12.00"),
        ("0012.00", "12.00"),
        ("999999999999.99", "999999999999.99"),  # MAX_INTEGER_DIGITS integer digits
    ],
)
def test_parse_amount_accepts(raw: str, expected: str) -> None:
    result = parse_amount(raw)

    assert result == Decimal(expected)
    assert str(result) == expected  # exactly two decimal places, no negative zero
    assert isinstance(result, Decimal)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "$",
        "-",
        "()",
        ".",
        "1.234,56",  # European format is ambiguous
        "1.234",  # three decimals (or a European thousands separator)
        "12.345",
        "12.",
        "1,23",  # malformed grouping
        "1,2345.00",
        ",123",
        "12 34",
        "NaN",
        "nan",
        "Infinity",
        "-Infinity",
        "inf",
        "1e3",
        "12.00 USD",
        "USD 12.00",
        "€12.00",
        "(-12.00)",  # conflicting negative notations
        "-(12.00)",
        "(12.00",
        "12.00)",
        "--12.00",
        "-$-12.00",
        "+-12.00",
        "12-",
        "abc",
        "١٢٣",  # Arabic-Indic digits
        "1" * (MAX_INTEGER_DIGITS + 1),
    ],
)
def test_parse_amount_rejects(raw: str) -> None:
    with pytest.raises(AmountParseError):
        parse_amount(raw)


def test_amount_parse_error_is_value_error() -> None:
    assert issubclass(AmountParseError, ValueError)


# ----------------------------------------------------------------------------- Money type


class _Holder(BaseModel):
    amount: Money


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (Decimal("12.3"), "12.30"),
        (Decimal("12.30"), "12.30"),
        (Decimal("12.300"), "12.30"),  # trailing zeros are not extra precision
        (Decimal("-0.00"), "0.00"),
        (12, "12.00"),
        ("12.30", "12.30"),
        ("-5", "-5.00"),
    ],
)
def test_money_accepts_and_quantizes(value: object, expected: str) -> None:
    holder = _Holder.model_validate({"amount": value})

    assert str(holder.amount) == expected


@pytest.mark.parametrize(
    "value",
    [
        12.30,  # float
        0.1,
        True,  # bool is an int subclass
        Decimal("12.345"),
        "12.345",
        Decimal("NaN"),
        Decimal("Infinity"),
        "NaN",
        "$12.00",  # formatted strings must go through parse_amount
        "1,234.00",
        Decimal("1" * (MAX_INTEGER_DIGITS + 1)),
    ],
)
def test_money_rejects(value: object) -> None:
    with pytest.raises(ValidationError):
        _Holder.model_validate({"amount": value})


def test_money_serializes_to_json_string_and_round_trips() -> None:
    holder = _Holder(amount=Decimal("1234.50"))

    json_text = holder.model_dump_json()

    assert json_text == '{"amount":"1234.50"}'
    assert _Holder.model_validate_json(json_text) == holder


def test_money_rejects_json_number_float() -> None:
    with pytest.raises(ValidationError):
        _Holder.model_validate_json('{"amount": 12.5}')


# ----------------------------------------------------------------------------- Decimal context

HOSTILE_CONTEXTS = [
    pytest.param(Context(prec=1), id="prec=1"),
    pytest.param(Context(prec=3, rounding=ROUND_DOWN), id="prec=3-round-down"),
    pytest.param(Context(prec=28, traps=[Inexact, Rounded]), id="trap-inexact-rounded"),
    pytest.param(Context(prec=2, rounding=ROUND_UP, traps=[Inexact, Rounded]), id="prec=2-traps"),
]


@pytest.mark.parametrize("context", HOSTILE_CONTEXTS)
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("$1,234.56", "1234.56"),
        ("(999,999,999,999.99)", "-999999999999.99"),
        ("12.5", "12.50"),
        ("-0.00", "0.00"),
    ],
)
def test_parse_amount_ignores_caller_decimal_context(
    context: Context, raw: str, expected: str
) -> None:
    with localcontext(context):
        result = parse_amount(raw)
        assert getcontext().prec == context.prec  # caller context left untouched

    assert str(result) == expected


@pytest.mark.parametrize("context", HOSTILE_CONTEXTS)
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("1234.56", "1234.56"),
        (Decimal("-999999999999.99"), "-999999999999.99"),
        (Decimal("12.300"), "12.30"),
        (7, "7.00"),
    ],
)
def test_money_validation_ignores_caller_decimal_context(
    context: Context, value: object, expected: str
) -> None:
    with localcontext(context):
        holder = _Holder.model_validate({"amount": value})
        json_text = holder.model_dump_json()

    assert str(holder.amount) == expected
    assert json_text == f'{{"amount":"{expected}"}}'


@pytest.mark.parametrize("context", HOSTILE_CONTEXTS)
@pytest.mark.parametrize(
    "value", ["12.345", "1E+999999", "1e-999999", Decimal("sNaN"), "sNaN", 1.5]
)
def test_money_rejections_are_validation_errors_in_any_context(
    context: Context, value: object
) -> None:
    with localcontext(context), pytest.raises(ValidationError):
        _Holder.model_validate({"amount": value})


def test_decimal_context_guard_translates_decimal_signals() -> None:
    def failing_handler(value: object) -> object:
        raise InvalidOperation("simulated")

    with pytest.raises(ValueError, match=r"invalid decimal value \(InvalidOperation\)"):
        decimal_context_guard("1", failing_handler)  # type: ignore[arg-type]


def test_decimal_context_guard_runs_handler_in_money_context() -> None:
    seen: list[int] = []

    def handler(value: object) -> object:
        seen.append(getcontext().prec)
        return value

    with localcontext(Context(prec=2)):
        decimal_context_guard("1", handler)  # type: ignore[arg-type]

    assert seen == [MONEY_CONTEXT.prec]

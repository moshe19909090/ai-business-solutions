from typing import Any

import pytest
from pydantic import ValidationError

from bookkeeping_assistant.profiles import (
    SUPPORTED_ACCOUNT_CURRENCIES,
    AccountProfile,
    CsvEncoding,
    DateOrder,
    NumberFormat,
    RegionalFormat,
)


def test_supported_account_currencies() -> None:
    assert frozenset({"USD", "ILS"}) == SUPPORTED_ACCOUNT_CURRENCIES


@pytest.mark.parametrize("currency", ["USD", "ILS"])
def test_account_profile_accepts_supported_currency(currency: str) -> None:
    assert AccountProfile(currency=currency).currency == currency


@pytest.mark.parametrize(
    ("currency", "message"),
    [
        ("EUR", "unsupported account currency; supported: ILS, USD"),
        ("NIS", "unsupported account currency"),
        ("ils", "String should match pattern"),
        ("₪", "String should match pattern"),
        ("", "String should match pattern"),
    ],
)
def test_account_profile_rejects_unsupported_currency(currency: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        AccountProfile(currency=currency)


def test_enum_values_are_stable() -> None:
    """Profile values appear in reports and API input. Changing them is a breaking change."""
    assert [d.value for d in DateOrder] == ["mdy", "dmy"]
    assert [e.value for e in CsvEncoding] == ["utf-8", "utf-16", "windows-1255"]
    assert [n.value for n in NumberFormat] == ["1,234.56"]


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"date_order": "ymd"}, id="date-order"),
        pytest.param({"csv_encoding": "latin-1"}, id="encoding"),
        pytest.param({"number_format": "1.234,56"}, id="number-format"),
        pytest.param({"language": "he"}, id="language-is-not-a-regional-format"),
    ],
)
def test_regional_format_rejects_unsupported(data: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        RegionalFormat.model_validate(data)


def test_profiles_are_frozen() -> None:
    profile = AccountProfile()

    with pytest.raises(ValidationError):
        profile.currency = "ILS"  # type: ignore[misc]

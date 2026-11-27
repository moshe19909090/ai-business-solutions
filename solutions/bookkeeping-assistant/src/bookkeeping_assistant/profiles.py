"""Account and regional-format profiles.

Three concerns are kept separate, and none is inferred from another:

- **Text and script** (`text.py`): how names in any writing system are cleaned and compared.
  Hebrew text says nothing about currency or date order.
- **Regional format** (`RegionalFormat`): how dates, numbers and CSV bytes are written in the
  client's files. It is declared by the user, never guessed from the data.
- **Account** (`AccountProfile`): the currency of the bank account. Every transaction in the CSV
  is in this currency, because the CSV has no currency column. Amounts in another currency are
  never compared with it, because there is no conversion.

Both profiles are part of `ReviewConfig` and are echoed in every review report.
"""

from enum import StrEnum
from typing import Final, Self

from pydantic import Field, model_validator

from bookkeeping_assistant.models import DomainModel

DEFAULT_ACCOUNT_CURRENCY: Final = "USD"
SUPPORTED_ACCOUNT_CURRENCIES: Final = frozenset({"USD", "ILS"})
"""Account currencies with demonstrated test coverage. Both use two decimal places, which is
what `Money` supports. Adding a currency means adding its aliases, fixtures and tests."""


class DateOrder(StrEnum):
    """How numeric dates such as ``05/03/2026`` are read. ISO ``2026-03-05`` is always accepted."""

    MDY = "mdy"
    """Month first (US): 05/03/2026 is 3 May 2026."""
    DMY = "dmy"
    """Day first (Israel, most of Europe): 05/03/2026 is 5 March 2026. Also allows 05.03.2026."""


class CsvEncoding(StrEnum):
    """Declared encoding of the transactions CSV. Single-byte code pages are never guessed."""

    UTF8 = "utf-8"
    """UTF-8, with or without a byte-order mark."""
    UTF16 = "utf-16"
    """UTF-16 with a byte-order mark, as written by some spreadsheet exports."""
    WINDOWS_1255 = "windows-1255"
    """Legacy Hebrew code page, used by some Israeli bank exports."""


class NumberFormat(StrEnum):
    """How amounts are written. Only one format is supported for now."""

    COMMA_GROUP_DOT_DECIMAL = "1,234.56"
    """US and Israeli format. Other formats (``1.234,56``) are rejected, never reinterpreted."""


class RegionalFormat(DomainModel):
    """How dates, numbers and CSV bytes are written in the client's files."""

    date_order: DateOrder = DateOrder.MDY
    csv_encoding: CsvEncoding = CsvEncoding.UTF8
    number_format: NumberFormat = NumberFormat.COMMA_GROUP_DOT_DECIMAL


class AccountProfile(DomainModel):
    """The bank account being reviewed."""

    currency: str = Field(
        default=DEFAULT_ACCOUNT_CURRENCY,
        pattern=r"^[A-Z]{3}$",
        description="ISO 4217 code of the account. Every transaction is in this currency.",
    )

    @model_validator(mode="after")
    def _currency_is_supported(self) -> Self:
        if self.currency not in SUPPORTED_ACCOUNT_CURRENCIES:
            supported = ", ".join(sorted(SUPPORTED_ACCOUNT_CURRENCIES))
            raise ValueError(f"unsupported account currency; supported: {supported}")
        return self

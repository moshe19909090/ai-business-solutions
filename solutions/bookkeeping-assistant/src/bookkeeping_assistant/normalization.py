"""Stage 3: deterministic normalization of extracted (untrusted) document fields.

`normalize_extraction` turns an `ExtractionResult` into a `NormalizedDocument`. It never raises
for bad content: every missing, invalid, unsupported, or ambiguous field becomes a
`NormalizationIssue`, so nothing is silently dropped.

Extractor output is untrusted. Text is cleaned of control and formatting characters (which
includes bidirectional overrides), and every value goes through strict, bounded parsers. Text
that looks like instructions ("ignore previous instructions...") is just data that fails to
parse.

`normalize_vendor` is also used for bank descriptions during matching (Phase 4), so both sides
are normalized the same way.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Final

from bookkeeping_assistant.extraction.schema import (
    DocumentExtraction,
    ExtractedField,
    ExtractionResult,
    ExtractionStatus,
)
from bookkeeping_assistant.ingestion import parse_csv_date
from bookkeeping_assistant.models import DocumentType, NormalizationIssue, NormalizedDocument
from bookkeeping_assistant.money import AmountParseError, parse_amount
from bookkeeping_assistant.profiles import DateOrder

SUPPORTED_CURRENCY: Final = "USD"
MAX_DATE_TEXT_LENGTH: Final = 40
MAX_DOCUMENT_NUMBER_LENGTH: Final = 64

_DOCUMENT_TYPES: Final = {
    "receipt": DocumentType.RECEIPT,
    "sales receipt": DocumentType.RECEIPT,
    "invoice": DocumentType.INVOICE,
    "bill": DocumentType.INVOICE,
    "other": DocumentType.OTHER,
    "unreadable": DocumentType.UNREADABLE,
}

# Unambiguous labels only. "$" (and "dollar") is shared by USD, CAD, AUD, MXN, ... so it is
# reported as ambiguous instead of being assumed to mean US dollars.
_CURRENCY_ALIASES: Final = {
    "USD": "USD",
    "US$": "USD",
    "US DOLLAR": "USD",
    "US DOLLARS": "USD",
    "€": "EUR",
    "EURO": "EUR",
    "EUROS": "EUR",
    "£": "GBP",
    "C$": "CAD",
    "CA$": "CAD",
    "A$": "AUD",
    "AU$": "AUD",
    "MX$": "MXN",
}
_AMBIGUOUS_CURRENCIES: Final = frozenset({"$", "DOLLAR", "DOLLARS"})
_ISO_CURRENCY_SHAPE: Final = re.compile(r"[A-Z]{3}", re.ASCII)

# "SQ *JOES COFFEE", "TST* BISTRO", "PAYPAL *STORE". The "*" is required, so "SPORTS" is safe.
_PROCESSOR_PREFIX: Final = re.compile(r"^(?:SQU?|TST|PAYPAL|PP|SP|PY|IN|IC)\s*\*\s*", re.ASCII)
# Numbers removed from vendor names only when clearly metadata: a complete "#" store number at
# the very end ("JOES COFFEE #0412", but not "THE #1 NAILS" or "STUDIO #54B"), and US phone
# numbers with separators. Phone numbers are removed first, so "... #0412 800-555-0100" works.
_STORE_NUMBER: Final = re.compile(r"(?<=\S)\s*+#\s*+[0-9]++\s*+$", re.ASCII)
_PHONE_NUMBER: Final = re.compile(
    r"(?<![0-9A-Z])(?:\+?1[-. ])?"
    r"(?:\([0-9]{3}\) ?|[0-9]{3}[-. ])[0-9]{3}[-. ][0-9]{4}"
    r"(?![0-9A-Z])",
    re.ASCII,
)
_LEGAL_SUFFIXES: Final = frozenset({"LLC", "INC", "CORP", "CORPORATION", "CO", "COMPANY", "LTD"})
_NON_ALPHANUMERIC: Final = re.compile(r"[^A-Z0-9]+", re.ASCII)

# An explicit US-dollar marker on an amount ("USD 12.40", "US$12.40", "12.40 USD"). Stripped
# before parsing; other currencies' labels are not, so those amounts stay invalid.
_AMOUNT_USD_MARKER: Final = re.compile(r"^(?:USD|US\$)\s*|\s*USD$", re.ASCII | re.IGNORECASE)

# Currency labels recognized on amounts, whether or not the amount parses. Conservative: only
# these unambiguous codes and symbols, and only at the very start or end of the amount text,
# allowing one leading "-" or "(" before a prefix label and one closing ")" after a suffix
# label ("-C$0.95", "(C$0.95)", "0.95 C$", "(0.95 C$)"). Labels inside words ("USDX", "XC$")
# do not count, and a bare "$" is never a label.
_AMOUNT_LABELS: Final = {
    "USD": "USD",
    "US$": "USD",
    "CAD": "CAD",
    "CA$": "CAD",
    "C$": "CAD",
    "AUD": "AUD",
    "AU$": "AUD",
    "A$": "AUD",
    "MXN": "MXN",
    "MX$": "MXN",
    "EUR": "EUR",
    "€": "EUR",
    "GBP": "GBP",
    "£": "GBP",
}
_LABEL_ALTERNATIVES: Final = r"USD|US\$|CAD|CA\$|C\$|AUD|AU\$|A\$|MXN|MX\$|EUR|GBP|€|£"
# Every quantifier is possessive or bounded, so malformed input is scanned in linear time.
_AMOUNT_LABEL: Final = re.compile(
    rf"^[-(]?\s*+({_LABEL_ALTERNATIVES})(?![A-Z])"
    rf"|(?<![A-Z])({_LABEL_ALTERNATIVES})\s*+\)?$",
    re.ASCII | re.IGNORECASE,
)
# Currencies written with a "$" sign, so consistent with a bare "$" in the currency field.
_DOLLAR_CURRENCIES: Final = frozenset({"USD", "CAD", "AUD", "MXN"})

# One separator ("T", a space, or a comma), then H:MM or H:MM:SS and an optional AM/PM.
# Ranges are checked in code. Inputs are capped at MAX_DATE_TEXT_LENGTH before searching.
_TIME_SUFFIX: Final = re.compile(
    r"(T| |, ?)([0-9]{1,2}):([0-9]{2})(?::([0-9]{2}))?(?: ?([AP])\.?M\.?)?$",
    re.ASCII | re.IGNORECASE,
)
_ISO_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", re.ASCII)
_SHORT_US_DATE: Final = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{2})", re.ASCII)
_MONTH_FIRST_DATE: Final = re.compile(r"([A-Za-z]{3,9})\.? ([0-9]{1,2}),? ([0-9]{4})", re.ASCII)
_DAY_FIRST_DATE: Final = re.compile(r"([0-9]{1,2})[ -]([A-Za-z]{3,9})\.?[ ,-]+([0-9]{4})", re.ASCII)
_MONTHS: Final = {
    name: number
    for number, names in enumerate(
        (
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        ),
        start=1,
    )
    for name in names
}


# ----------------------------------------------------------------------------- text


def clean_text(value: str | None) -> str | None:
    """Remove control and formatting characters, collapse whitespace. Returns None if empty.

    Control characters (Cc: tabs, newlines, NUL) become spaces. Other invisible characters
    (Cf, Co, Cs, Cn: zero-width and bidirectional overrides, private use) are removed.
    """
    if value is None:
        return None
    kept = []
    for char in value:
        category = unicodedata.category(char)
        if category == "Cc":
            kept.append(" ")
        elif not category.startswith("C"):
            kept.append(char)
    collapsed = " ".join("".join(kept).split())
    return collapsed or None


def normalize_vendor(text: str) -> str:
    """Canonical vendor key for matching. Returns "" if nothing meaningful remains.

    Folds to uppercase ASCII and strips a leading payment-processor prefix (``SQ *``, ``TST*``,
    ``PAYPAL *``, ...). Removes numbers only when they are clearly metadata: a complete ``#``
    store number at the end of the name, and US phone numbers written with separators. Other
    numbers are part of the name and are kept (``STUDIO 54``, ``THE #1 NAILS``, ``STUDIO #54B``,
    ``76``). Drops apostrophes, turns other punctuation into spaces, and finally drops trailing
    legal suffixes (LLC, INC, CO, ...).

    Examples: ``"SQ *JOE'S COFFEE #0412"`` → ``"JOES COFFEE"``;
    ``"Office Supply Depot, Inc."`` → ``"OFFICE SUPPLY DEPOT"``.
    """
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    folded = folded.upper().strip()
    folded = _PROCESSOR_PREFIX.sub("", folded, count=1)
    folded = _STORE_NUMBER.sub(" ", _PHONE_NUMBER.sub(" ", folded))
    folded = _NON_ALPHANUMERIC.sub(" ", folded.replace("'", "").replace("`", ""))
    tokens = folded.split()
    while len(tokens) > 1 and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


# ----------------------------------------------------------------------------- dates


@dataclass(frozen=True, slots=True)
class ParsedDate:
    value: date
    ambiguous: bool
    """True when an assumption was needed, e.g. a two-digit year (assumed 20xx)."""


def parse_document_date(text: str) -> ParsedDate:
    """Parse a date as printed on a receipt or invoice. Raises `ValueError` if not possible.

    Accepted:
    - ``YYYY-MM-DD`` and ``M/D/YYYY`` (US month-first), as in the transactions CSV
    - ``M/D/YY``: the year is assumed to be 20YY and the result is marked ambiguous
    - ``Mar 4, 2026``, ``March 4 2026``, ``4 Mar 2026``, ``04-Mar-2026``

    An optional trailing time is validated, then ignored: ``08:14``, ``8:14:55``, ``8:14 PM``
    after a space or comma, or ``T08:14:55`` after an ISO date (24-hour only). Hours, minutes,
    and seconds must be in range (1-12 with AM/PM, otherwise 0-23), so a malformed timestamp
    rejects the whole date instead of being discarded.

    Month names are matched in English without using the system locale, so results do not
    depend on the machine's settings. Day-first numeric dates (``13/04/2026``) are rejected.
    """
    collapsed = " ".join(text.split())
    if not collapsed or len(collapsed) > MAX_DATE_TEXT_LENGTH:
        raise ValueError("invalid date: empty or too long")
    cleaned = _strip_time(collapsed)

    try:
        return ParsedDate(parse_csv_date(cleaned), ambiguous=False)
    except ValueError:
        pass  # not a CSV-style date; try the receipt formats below
    if match := _SHORT_US_DATE.fullmatch(cleaned):
        month, day, year = (int(part) for part in match.groups())
        return ParsedDate(_calendar_date(2000 + year, month, day), ambiguous=True)
    if match := _MONTH_FIRST_DATE.fullmatch(cleaned):
        month_text, day_text, year_text = match.groups()
        value = _calendar_date(int(year_text), _month(month_text), int(day_text))
        return ParsedDate(value, ambiguous=False)
    if match := _DAY_FIRST_DATE.fullmatch(cleaned):
        day_text, month_text, year_text = match.groups()
        value = _calendar_date(int(year_text), _month(month_text), int(day_text))
        return ParsedDate(value, ambiguous=False)
    raise ValueError("invalid date: unrecognized format")


def _strip_time(text: str) -> str:
    """Return the date part of ``text`` after validating any trailing time."""
    match = _TIME_SUFFIX.search(text)
    if match is None:
        return text
    separator, hour_text, minute_text, second_text, meridiem = match.groups()
    date_part = text[: match.start()]
    hour, minute = int(hour_text), int(minute_text)
    second = int(second_text) if second_text is not None else 0
    doubled_separator = date_part.endswith((" ", ","))  # "2026-03-02 ,08:14"
    is_t = separator in ("T", "t")  # the pattern is case-insensitive, so check both cases
    misplaced_t = is_t and (meridiem is not None or not _ISO_DATE.fullmatch(date_part))
    if doubled_separator or misplaced_t:
        raise ValueError("invalid date: malformed time")
    hour_ok = 1 <= hour <= 12 if meridiem is not None else 0 <= hour <= 23
    if not (hour_ok and minute <= 59 and second <= 59):
        raise ValueError("invalid date: time out of range")
    return date_part


def _month(name: str) -> int:
    number = _MONTHS.get(name.lower())
    if number is None:
        raise ValueError("invalid date: unrecognized month name")
    return number


def _calendar_date(year: int, month: int, day: int) -> date:
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError("invalid date: not a real calendar date") from exc


# ----------------------------------------------------------------------------- currency


class AmbiguousCurrencyError(ValueError):
    """The text names a symbol shared by several currencies, such as ``$``."""


def normalize_currency(text: str) -> str:
    """Map a printed currency to an ISO-4217-style code. Raises `ValueError` if unrecognized.

    Unambiguous aliases and symbols are mapped (``US$`` → USD, ``C$`` → CAD, ``€`` → EUR, ...).
    A bare ``$`` or "dollars" raises `AmbiguousCurrencyError` (a `ValueError`): it is never
    assumed to mean USD. Any other three-letter code is returned uppercased. It is shape-checked
    only, not against the full ISO list.
    """
    key = " ".join(text.upper().split())
    if key in _AMBIGUOUS_CURRENCIES:
        raise AmbiguousCurrencyError("ambiguous currency: symbol shared by several currencies")
    if key in _CURRENCY_ALIASES:
        return _CURRENCY_ALIASES[key]
    if _ISO_CURRENCY_SHAPE.fullmatch(key):
        return key
    raise ValueError("invalid currency: unrecognized")


# ----------------------------------------------------------------------------- documents


def normalize_extraction(
    result: ExtractionResult, *, date_order: DateOrder = DateOrder.MDY
) -> NormalizedDocument:
    """Normalize one extraction result. Never raises for bad extracted content.

    `date_order` is the declared order for numeric dates (``RegionalFormat.date_order``).
    """
    if date_order is not DateOrder.MDY:  # contract stub: day-first support lands in the i18n batch
        raise NotImplementedError("date_order other than mdy is not supported yet")
    if result.status is not ExtractionStatus.OK or result.extraction is None:
        return NormalizedDocument(
            document_id=result.document_id,
            document_type=DocumentType.UNREADABLE,
            issues=(NormalizationIssue.EXTRACTION_FAILED,),
        )
    return _Normalizer(result.extraction).build(result.document_id)


class _Normalizer:
    """Collects issues while normalizing each field of one extraction (single use)."""

    def __init__(self, extraction: DocumentExtraction) -> None:
        self._extraction = extraction
        self._issues: list[NormalizationIssue] = []
        self._amount_currencies: set[str] = set()  # labels seen on amounts, parsed or not

    def build(self, document_id: str) -> NormalizedDocument:
        extraction = self._extraction
        document_type = self._document_type(extraction.document_type)
        vendor, vendor_normalized = self._vendor(extraction.vendor_name)
        document_date = self._date(extraction.document_date)
        total = self._money(
            extraction.total, NormalizationIssue.TOTAL_INVALID, NormalizationIssue.TOTAL_MISSING
        )
        if total is not None and total <= 0:
            self._issue(NormalizationIssue.TOTAL_NOT_POSITIVE)
        subtotal = self._money(extraction.subtotal, NormalizationIssue.SUBTOTAL_INVALID)
        tax = self._money(extraction.tax, NormalizationIssue.TAX_INVALID)
        tip = self._money(extraction.tip, NormalizationIssue.TIP_INVALID)
        return NormalizedDocument(
            document_id=document_id,
            document_type=document_type,
            vendor=vendor,
            vendor_normalized=vendor_normalized,
            document_date=document_date,
            total=total,
            subtotal=subtotal,
            tax=tax,
            tip=tip,
            currency=self._currency(extraction.currency),
            document_number=self._document_number(extraction.document_number),
            issues=tuple(self._issues),
        )

    def _issue(self, issue: NormalizationIssue) -> None:
        if issue not in self._issues:
            self._issues.append(issue)

    def _document_type(self, field: ExtractedField | None) -> DocumentType:
        text = _value(field)
        if text is None:
            self._issue(NormalizationIssue.DOCUMENT_TYPE_MISSING)
            return DocumentType.OTHER
        document_type = _DOCUMENT_TYPES.get(text.lower())
        if document_type is None:
            self._issue(NormalizationIssue.DOCUMENT_TYPE_UNRECOGNIZED)
            return DocumentType.OTHER
        if document_type is DocumentType.UNREADABLE:
            self._issue(NormalizationIssue.DOCUMENT_UNREADABLE)
        return document_type

    def _vendor(self, field: ExtractedField | None) -> tuple[str | None, str | None]:
        vendor = _value(field)
        if vendor is None:
            self._issue(NormalizationIssue.VENDOR_MISSING)
            return None, None
        normalized = normalize_vendor(vendor)
        if not normalized:
            self._issue(NormalizationIssue.VENDOR_INVALID)
            return vendor, None
        return vendor, normalized

    def _date(self, field: ExtractedField | None) -> date | None:
        text = _value(field)
        if text is None:
            self._issue(NormalizationIssue.DATE_MISSING)
            return None
        try:
            parsed = parse_document_date(text)
        except ValueError:
            self._issue(NormalizationIssue.DATE_INVALID)
            return None
        if parsed.ambiguous:
            self._issue(NormalizationIssue.DATE_AMBIGUOUS)
        return parsed.value

    def _money(
        self,
        field: ExtractedField | None,
        invalid: NormalizationIssue,
        missing: NormalizationIssue | None = None,
    ) -> Decimal | None:
        text = _value(field)
        if text is None:
            if missing is not None:
                self._issue(missing)
            return None
        # Currency evidence is recorded before parsing, so a label on an amount that fails to
        # parse ("C$0.95") still counts. A bare "$" says nothing about which dollar it is.
        self._amount_currencies.update(
            _AMOUNT_LABELS[(match.group(1) or match.group(2)).upper()]
            for match in _AMOUNT_LABEL.finditer(text)
        )
        try:
            return parse_amount(_AMOUNT_USD_MARKER.sub("", text))
        except AmountParseError:
            self._issue(invalid)
            return None

    def _currency(self, field: ExtractedField | None) -> str | None:
        """Resolve the currency from the currency field and the currency labels on amounts.

        Must run after the amounts. A missing or "$"-only currency field is resolved only by a
        single consistent label; any disagreement leaves the currency unset as ambiguous.
        """
        labels = self._amount_currencies
        text = _value(field)
        if text is None:
            if not labels:
                self._issue(NormalizationIssue.CURRENCY_MISSING)
                return None
            return self._single_label(labels)
        try:
            currency = normalize_currency(text)
        except AmbiguousCurrencyError:
            # A "$" is resolved only by a single dollar-sign label; "€" contradicts it.
            if len(labels) == 1 and labels <= _DOLLAR_CURRENCIES:
                return self._single_label(labels)
            self._issue(NormalizationIssue.CURRENCY_AMBIGUOUS)
            return None
        except ValueError:
            self._issue(NormalizationIssue.CURRENCY_INVALID)
            return None
        return self._single_label(labels | {currency})

    def _single_label(self, currencies: set[str]) -> str | None:
        """The one currency all evidence agrees on; otherwise unset and flagged ambiguous."""
        if len(currencies) != 1:
            self._issue(NormalizationIssue.CURRENCY_AMBIGUOUS)
            return None
        (currency,) = currencies
        if currency != SUPPORTED_CURRENCY:
            self._issue(NormalizationIssue.CURRENCY_UNSUPPORTED)
        return currency

    def _document_number(self, field: ExtractedField | None) -> str | None:
        text = _value(field)
        if text is None:
            return None  # optional: most receipts have no number worth comparing
        if len(text) > MAX_DOCUMENT_NUMBER_LENGTH:
            self._issue(NormalizationIssue.DOCUMENT_NUMBER_INVALID)
            return None
        return text


def _value(field: ExtractedField | None) -> str | None:
    """The cleaned field value. A null field, null value, or blank text all count as missing."""
    return clean_text(field.value) if field is not None else None

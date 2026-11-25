"""Strict monetary parsing and the `Money` type.

Rules:
- Money is always `Decimal` with exactly two decimal places (USD cents). It is never `float`.
- Parsing is strict and US-formatted. Ambiguous or locale-specific inputs (for example
  `1.234,56`) are rejected instead of guessed.
- Decimal arithmetic runs in a fixed local context (`MONEY_CONTEXT`), so results never depend
  on the caller's thread-local Decimal context (precision, rounding, traps).
"""

import re
from collections.abc import Iterator
from contextlib import contextmanager
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from typing import Annotated, Final

from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)

MONEY_CONTEXT: Final = Context(
    prec=28,
    rounding=ROUND_HALF_EVEN,
    traps=[InvalidOperation, DivisionByZero, Overflow],
)
"""Template context for all money arithmetic. `localcontext` copies it, so it is never mutated."""

CENT: Final = Decimal("0.01")
MAX_INTEGER_DIGITS: Final = 12
"""Upper bound on the integer part (below one trillion). Guards against absurd or garbled values."""

MAX_AMOUNT_TEXT_LENGTH: Final = 64
"""Longest accepted amount text after trimming. The longest valid amount,
``"($ 999,999,999,999.99 )"``, is far shorter. This is defense in depth on top of the
linear-time pattern below."""

# Possessive quantifiers (`*+`, `++`) never give back what they matched. With plain `\s*`, the
# four whitespace runs are separated only by optional tokens, so a long whitespace run before an
# invalid character could be split between them in O(n^4) ways before failing. Possessive
# matching is equivalent here, because the element after each run can never be whitespace or a
# digit, so backtracking could never produce a different successful match. Matching is O(n).
_AMOUNT_PATTERN: Final = re.compile(
    r"""
    ^
    (?P<open>\()?\s*+                                 # accounting negative: (12.00)
    (?P<sign>[-+])?\s*+                               # sign before the currency symbol
    \$?\s*+                                           # optional dollar sign
    (?P<sign_after_symbol>[-+])?\s*+                  # sign after the currency symbol: $-12.00
    (?P<integer>[0-9]{1,3}(?:,[0-9]{3})++|[0-9]*+)    # 1,234,567 or 1234567 (or empty for .50)
    (?:\.(?P<fraction>[0-9]{1,2}))?                   # at most two decimal places
    \s*+(?P<close>\))?
    $
    """,
    re.VERBOSE | re.ASCII,  # ASCII: reject non-ASCII digits such as Arabic-Indic numerals
)


class AmountParseError(ValueError):
    """Raised when a string cannot be parsed unambiguously into a monetary amount."""


def parse_amount(raw: str) -> Decimal:
    """Parse a US-formatted amount string into a `Decimal` with two decimal places.

    Accepted examples: ``"12"``, ``"12.5"``, ``"-12.00"``, ``"$1,234.56"``, ``"(12.00)"``,
    ``"-$12.00"``, ``"$-12.00"``, ``"$.50"``.

    Rejected examples: ``""``, ``"1.234,56"``, ``"12.345"``, ``"1,23"``, ``"NaN"``,
    ``"Infinity"``, ``"(-12.00)"``, ``"12 34"``.

    Raises:
        AmountParseError: if the input is empty, malformed, ambiguous, or out of range.
    """
    text = raw.strip()
    if not text:
        raise AmountParseError("amount is empty")
    if len(text) > MAX_AMOUNT_TEXT_LENGTH:
        raise AmountParseError(f"amount text exceeds {MAX_AMOUNT_TEXT_LENGTH} characters")

    match = _AMOUNT_PATTERN.match(text)
    if match is None:
        raise AmountParseError(f"unrecognized amount format: {text!r}")

    has_open, has_close = match["open"] is not None, match["close"] is not None
    if has_open != has_close:
        raise AmountParseError(f"unbalanced parentheses: {text!r}")

    signs = [s for s in (match["sign"], match["sign_after_symbol"]) if s is not None]
    if len(signs) > 1:
        raise AmountParseError(f"multiple signs: {text!r}")
    if has_open and signs:
        raise AmountParseError(f"conflicting negative notations: {text!r}")

    integer_digits = match["integer"].replace(",", "")
    fraction = match["fraction"]
    if not integer_digits and fraction is None:
        raise AmountParseError(f"amount has no digits: {text!r}")
    if len(integer_digits.lstrip("0")) > MAX_INTEGER_DIGITS:
        raise AmountParseError(f"amount exceeds {MAX_INTEGER_DIGITS} integer digits: {text!r}")

    is_negative = has_open or signs == ["-"]
    try:
        value = _canonical(Decimal(f"{integer_digits or '0'}.{fraction or '0'}"))
    except DecimalException as exc:  # defensive: unreachable with MONEY_CONTEXT and digit limit
        raise AmountParseError(f"amount cannot be represented: {text!r}") from exc
    return value.copy_negate() if is_negative and not value.is_zero() else value


@contextmanager
def _money_context() -> Iterator[None]:
    with localcontext(MONEY_CONTEXT):
        yield


def _canonical(value: Decimal) -> Decimal:
    """Quantize to cents and collapse negative zero, so ``-0.00`` serializes as ``0.00``."""
    with _money_context():
        quantized = value.quantize(CENT)
    # copy_abs is exact and context-free, unlike abs().
    return quantized.copy_abs() if quantized.is_zero() else quantized


def decimal_context_guard(value: object, handler: ValidatorFunctionWrapHandler) -> object:
    """Pydantic wrap validator: run inner Decimal validation in `MONEY_CONTEXT`.

    Pydantic's own Decimal parsing and constraint checks use the thread-local context. Running
    them inside `MONEY_CONTEXT` makes validation independent of the caller. Any Decimal signal
    is translated into a `ValueError`, which Pydantic reports as a `ValidationError`.
    """
    with _money_context():
        try:
            return handler(value)
        except DecimalException as exc:
            raise ValueError(f"invalid decimal value ({type(exc).__name__})") from exc


def reject_float(value: object) -> object:
    # bool is a subclass of int, so reject it explicitly.
    if isinstance(value, float | bool):
        raise ValueError("monetary values must be a Decimal, int, or numeric string, not float")
    return value


def _validate_money(value: Decimal) -> Decimal:
    if not value.is_finite():
        raise ValueError("monetary values must be finite")
    canonical = _canonical(value)
    if canonical != value:
        raise ValueError("monetary values must have at most 2 decimal places")
    return canonical


# Annotated metadata applies inside-out: the last entry (the context guard) wraps everything.
Money = Annotated[
    Decimal,
    BeforeValidator(reject_float),
    Field(max_digits=MAX_INTEGER_DIGITS + 2, decimal_places=2, allow_inf_nan=False),
    AfterValidator(_validate_money),
    WrapValidator(decimal_context_guard),
]
"""A USD amount with exactly two decimal places. Serializes to JSON as a string (e.g. ``"12.30"``).

Accepts `Decimal`, `int`, or plain numeric strings (``"12.30"``). It rejects `float`, `bool`,
NaN, infinities, and more than two decimal places. Formatted strings such as ``"$1,234.56"``
must go through `parse_amount` explicitly.
"""

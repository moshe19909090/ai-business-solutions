"""Review configuration: thresholds, date windows, and scoring weights.

Every value here is a heuristic default, not an accounting, audit, or tax standard. The full
configuration is echoed in each review report so findings can always be traced to the
parameters that produced them.

Environment-based settings (API keys, extractor selection) are added in Phase 6.
"""

import math
from decimal import Decimal
from typing import Annotated, Self

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, WrapValidator, model_validator

from bookkeeping_assistant.models import SCORE_TOLERANCE, Score, reject_bool
from bookkeeping_assistant.money import Money, decimal_context_guard, reject_float

NonNegativeMoney = Annotated[Money, Field(ge=0)]
Ratio = Annotated[
    Decimal,
    BeforeValidator(reject_float),
    Field(ge=0, le=1, allow_inf_nan=False),
    WrapValidator(decimal_context_guard),
]
"""A fraction in [0, 1] that is compared against money amounts, so it is a `Decimal`."""
Weight = Score
"""Same bounds, bool rejection, and rounding tolerance as the scores the weights produce."""
WindowDays = Annotated[int, BeforeValidator(reject_bool), Field(ge=0, le=366)]


class _Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ScoreWeights(_Config):
    """Weights of the match score components. A perfect match scores exactly 1.0."""

    amount_exact: Weight = 0.5
    amount_close: Weight = 0.25
    date: Weight = 0.2
    vendor: Weight = 0.3

    @model_validator(mode="after")
    def _check_weights(self) -> Self:
        maximum = self.amount_exact + self.date + self.vendor
        if not math.isclose(maximum, 1.0, rel_tol=0.0, abs_tol=SCORE_TOLERANCE):
            raise ValueError(f"amount_exact + date + vendor must equal 1.0 (got {maximum})")
        if self.amount_close > self.amount_exact:
            raise ValueError("amount_close must not exceed amount_exact")
        return self


class ReviewConfig(_Config):
    """Tunable parameters for matching and validation."""

    # Missing documents
    missing_document_threshold: NonNegativeMoney = Field(
        default=Decimal("75.00"),
        description="Outflows at or above this amount without a document are flagged.",
    )

    # Date windows: how far a document date may fall before or after the posted date
    receipt_days_before: WindowDays = 5
    receipt_days_after: WindowDays = 2
    invoice_days_before: WindowDays = 60
    invoice_days_after: WindowDays = 2

    # Amount comparison (money-facing, so Decimal)
    amount_close_ratio: Ratio = Field(
        default=Decimal("0.05"),
        description="Relative difference still scored as a 'close' amount.",
    )
    tip_hint_ratio: Ratio = Field(
        default=Decimal("0.20"),
        description="Transaction above the document total by up to this ratio gets a tip hint.",
    )
    high_severity_delta: NonNegativeMoney = Decimal("50.00")
    high_severity_ratio: Ratio = Decimal("0.10")
    arithmetic_tolerance: NonNegativeMoney = Field(
        default=Decimal("0.01"),
        description="Allowed difference for the subtotal + tax + tip = total check.",
    )

    # Matching heuristics (scores, so float)
    vendor_min_similarity: Weight = 0.6
    ambiguity_margin: Weight = 0.05
    weights: ScoreWeights = Field(default_factory=ScoreWeights)

"""Core domain models shared by the pipeline stages.

All models are immutable (`frozen`) and reject unknown fields. Collections are tuples and
string maps are read-only `MappingProxyType`s, so validated instances are deeply immutable and
safe to share between stages. (`model_construct` and `model_copy(update=...)` skip validation
and therefore also skip freezing. Pipeline code must not use them for these models.)

Deferred by design:
- `ExtractionResult` (Phase 3) depends on the LLM extraction schema.
- `ReviewReport` (Phase 5) depends on `ExtractionResult`.
"""

import datetime as dt
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Annotated, Final, Literal, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    model_validator,
)

from bookkeeping_assistant.money import Money

_SHA256_PATTERN: Final = r"^[0-9a-f]{64}$"
DOCUMENT_ID_LENGTH: Final = 16

SCORE_TOLERANCE: Final = 1e-9
"""Float rounding slack for scores and weights. Sums such as 0.33 + 0.56 + 0.11 evaluate to
1.0000000000000002. Anything within this tolerance of [0, 1] is clamped. Anything further
outside is rejected."""


def reject_bool(value: object) -> object:
    """Reject `bool` for numeric fields. Pydantic's lax mode would coerce True to 1."""
    if isinstance(value, bool):
        raise ValueError("boolean values are not accepted for numeric fields")
    return value


def _bounded_score(value: float) -> float:
    if value < -SCORE_TOLERANCE or value > 1.0 + SCORE_TOLERANCE:
        raise ValueError(f"score must be within [0, 1], got {value}")
    return min(max(value, 0.0), 1.0)


def _freeze_str_map(value: Mapping[str, str]) -> Mapping[str, str]:
    # Copy first, so later changes to the caller's dict cannot leak into the model.
    return MappingProxyType(dict(value))


def _thaw_str_map(value: Mapping[str, str]) -> dict[str, str]:
    return dict(value)


NonEmptyStr = Annotated[str, Field(min_length=1)]
Score = Annotated[
    float,
    BeforeValidator(reject_bool),
    Field(allow_inf_nan=False),
    AfterValidator(_bounded_score),
]
"""Heuristic score in [0, 1] (with `SCORE_TOLERANCE`). Scores are not money, so `float` is fine."""
FrozenStrMap = Annotated[
    Mapping[str, str],
    AfterValidator(_freeze_str_map),
    PlainSerializer(_thaw_str_map, return_type=dict[str, str]),
]
"""A read-only `str -> str` mapping. Serializes as a plain JSON object."""


class DomainModel(BaseModel):
    """Base for all domain models: immutable, strict about unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


# ----------------------------------------------------------------------------- enums


class Direction(StrEnum):
    OUTFLOW = "outflow"
    INFLOW = "inflow"
    NEUTRAL = "neutral"


class MediaType(StrEnum):
    PDF = "application/pdf"
    PNG = "image/png"
    JPEG = "image/jpeg"


class DocumentType(StrEnum):
    RECEIPT = "receipt"
    INVOICE = "invoice"
    OTHER = "other"
    UNREADABLE = "unreadable"


class MatchStatus(StrEnum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"


class AmbiguityReason(StrEnum):
    MULTIPLE_DOCUMENTS = "multiple_documents"
    """This transaction has two or more near-tied candidate documents."""
    SHARED_DOCUMENT = "shared_document"
    """A candidate document is near-tied between this transaction and at least one other."""


class FindingType(StrEnum):
    MISSING_DOCUMENT = "missing_document"
    AMOUNT_MISMATCH = "amount_mismatch"
    DUPLICATE_DOCUMENT = "duplicate_document"
    AMBIGUOUS_MATCH = "ambiguous_match"
    INCOMPLETE_DOCUMENT = "incomplete_document"
    UNMATCHED_DOCUMENT = "unmatched_document"


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Provenance(StrEnum):
    """Where a piece of evidence came from, so reviewers can tell AI output from facts."""

    BANK_EXPORT = "bank_export"
    AI_EXTRACTION = "ai_extraction"
    COMPUTED = "computed"


# ----------------------------------------------------------------------------- inputs


class BankTransaction(DomainModel):
    """One row of the bank-transaction export.

    `amount` is signed as in the source: negative means money left the account.
    """

    id: NonEmptyStr
    row_number: int = Field(ge=1, description="1-based line number in the source CSV.")
    posted_date: dt.date
    description: NonEmptyStr
    amount: Money
    raw: FrozenStrMap = Field(description="Original CSV row values, verbatim, for evidence.")

    @property
    def direction(self) -> Direction:
        """Derived from the sign, so it can never disagree with `amount`."""
        if self.amount < 0:
            return Direction.OUTFLOW
        if self.amount > 0:
            return Direction.INFLOW
        return Direction.NEUTRAL


class SourceDocument(DomainModel):
    """An uploaded supporting document (receipt, invoice, etc.), identified by content hash."""

    id: NonEmptyStr = Field(description="First 16 hex characters of `sha256`.")
    filename: NonEmptyStr = Field(description="Sanitized display name. Never used as a path.")
    media_type: MediaType
    size_bytes: int = Field(ge=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)

    @model_validator(mode="after")
    def _id_derives_from_hash(self) -> Self:
        if self.id != self.sha256[:DOCUMENT_ID_LENGTH]:
            raise ValueError("id must equal the first 16 characters of sha256")
        return self


class NormalizedDocument(DomainModel):
    """Deterministically normalized view of a document's extracted fields."""

    document_id: NonEmptyStr
    document_type: DocumentType
    vendor: str | None = None
    vendor_normalized: str | None = None
    document_date: dt.date | None = None
    total: Money | None = None
    subtotal: Money | None = None
    tax: Money | None = None
    tip: Money | None = None
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    document_number: str | None = None
    issues: tuple[str, ...] = Field(
        default=(), description="Normalization problems, e.g. unparseable total."
    )


# ----------------------------------------------------------------------------- matching


class ScoreBreakdown(DomainModel):
    """Every component of a match score, so the score can be explained and audited."""

    amount_delta: Money = Field(description="abs(transaction amount) minus document total.")
    amount_component: Score
    date_delta_days: int = Field(description="Posted date minus document date, in days.")
    date_component: Score
    vendor_similarity: Score
    vendor_component: Score
    total: Score


class MatchCandidate(DomainModel):
    transaction_id: NonEmptyStr
    document_id: NonEmptyStr
    breakdown: ScoreBreakdown


class MatchResult(DomainModel):
    """Matching outcome for one transaction.

    Ambiguity has two causes, and both can apply at once:
    - `MULTIPLE_DOCUMENTS`: this transaction has at least two near-tied `candidates`.
    - `SHARED_DOCUMENT`: one of its candidate documents is near-tied with another transaction.
      `competing_candidates` lists those other transactions' candidates for the shared
      document(s), with their score breakdowns, so the contention can be explained. In this
      case a single `candidate` is enough.
    """

    transaction_id: NonEmptyStr
    status: MatchStatus
    document_id: str | None = None
    candidates: tuple[MatchCandidate, ...] = ()
    ambiguity_reasons: tuple[AmbiguityReason, ...] = ()
    competing_candidates: tuple[MatchCandidate, ...] = Field(
        default=(),
        description="Other transactions' candidates for this transaction's contested documents.",
    )

    @model_validator(mode="after")
    def _status_is_consistent(self) -> Self:
        self._check_candidates()
        if self.status is MatchStatus.AMBIGUOUS:
            self._check_ambiguity()
            return self
        if self.ambiguity_reasons or self.competing_candidates:
            raise ValueError(
                "ambiguity_reasons and competing_candidates are only allowed when ambiguous"
            )
        if self.status is MatchStatus.MATCHED:
            if self.document_id is None or self.document_id not in self._candidate_document_ids:
                raise ValueError("a matched result must reference one of its candidates")
        elif self.document_id is not None or self.candidates:
            raise ValueError("an unmatched result must have no document_id and no candidates")
        return self

    @property
    def _candidate_document_ids(self) -> set[str]:
        return {c.document_id for c in self.candidates}

    def _check_candidates(self) -> None:
        if any(c.transaction_id != self.transaction_id for c in self.candidates):
            raise ValueError("all candidates must belong to this transaction")
        if len(self._candidate_document_ids) != len(self.candidates):
            raise ValueError("candidates must reference distinct documents")
        pairs = {(c.transaction_id, c.document_id) for c in self.competing_candidates}
        if len(pairs) != len(self.competing_candidates):
            raise ValueError("competing_candidates must not contain duplicates")
        for competitor in self.competing_candidates:
            if competitor.transaction_id == self.transaction_id:
                raise ValueError("competing_candidates must belong to other transactions")
            if competitor.document_id not in self._candidate_document_ids:
                raise ValueError("competing_candidates must reference this result's candidates")

    def _check_ambiguity(self) -> None:
        if self.document_id is not None:
            raise ValueError("document_id must be empty when status is ambiguous")
        if not self.ambiguity_reasons:
            raise ValueError("an ambiguous result must state at least one ambiguity reason")
        if len(set(self.ambiguity_reasons)) != len(self.ambiguity_reasons):
            raise ValueError("ambiguity_reasons must not contain duplicates")
        if not self.candidates:
            raise ValueError("an ambiguous result must list at least one candidate")
        if (
            AmbiguityReason.MULTIPLE_DOCUMENTS in self.ambiguity_reasons
            and len(self.candidates) < 2
        ):
            raise ValueError("multiple_documents ambiguity requires at least two candidates")
        shared = AmbiguityReason.SHARED_DOCUMENT in self.ambiguity_reasons
        if shared != bool(self.competing_candidates):
            raise ValueError(
                "shared_document ambiguity requires competing_candidates, and vice versa"
            )


# ----------------------------------------------------------------------------- findings


class Evidence(DomainModel):
    """One traceable fact supporting a finding."""

    provenance: Provenance
    label: NonEmptyStr = Field(description="Human-readable name, e.g. 'Document total'.")
    value: str = Field(description="The value as shown to the reviewer.")
    source_ref: NonEmptyStr = Field(description="Locator, e.g. 'csv:row:14' or 'doc:<id>:total'.")


class RuleRef(DomainModel):
    """The deterministic rule that produced a finding, including the parameters used."""

    rule_id: NonEmptyStr
    rule_version: NonEmptyStr
    # validate_default so the default empty map is frozen too.
    parameters: FrozenStrMap = Field(default_factory=dict, validate_default=True)


class Finding(DomainModel):
    """An exception that needs human review. Never an accounting conclusion."""

    id: NonEmptyStr
    type: FindingType
    severity: Severity
    status: Literal["needs_review"] = "needs_review"
    title: NonEmptyStr
    explanation: NonEmptyStr
    transaction_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    rule: RuleRef

    @model_validator(mode="after")
    def _references_something(self) -> Self:
        if not self.transaction_ids and not self.document_ids:
            raise ValueError("a finding must reference at least one transaction or document")
        return self

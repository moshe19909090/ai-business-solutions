"""Core domain models shared by the pipeline stages.

All models are immutable (`frozen`) and reject unknown fields. Collections are tuples and
string maps are read-only `MappingProxyType`s, so validated instances are deeply immutable and
safe to share between stages. (`model_construct` and `model_copy(update=...)` skip validation
and therefore also skip freezing. Pipeline code must not use them for these models.)

Defined elsewhere to avoid import cycles:
- `ExtractionResult` lives in `extraction/schema.py`, which imports this module.
- `ReviewReport` lives in `review.py`, because it contains `ExtractionResult`.

Reviewer-facing text is never stored as the source of truth. Findings and evidence carry stable
codes and locale-neutral parameters; titles, explanations and labels are rendered from a message
catalog (`messages.py`) in the report's locale.
"""

import datetime as dt
import re
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


_CODE_PATTERN: Final = re.compile(r"[a-z][a-z0-9_]*")
_PARAM_VALUE_PATTERN: Final = re.compile(r"[A-Za-z0-9_.:+\-]{1,64}")
MAX_MESSAGE_PARAMS: Final = 16


def _freeze_message_params(value: Mapping[str, str]) -> Mapping[str, str]:
    """Message parameters are locale-neutral tokens only, never free text.

    Keys are snake_case codes. Values are short ASCII tokens: amounts (``-12.40``), ISO currency
    codes, ISO dates, counts, ratios, IDs and codes. Vendor names, descriptions, filenames and
    quotes cannot pass, so they can only appear in evidence. This keeps rendered sentences free
    of mixed-direction text and of prompt-injection text from documents.
    """
    if len(value) > MAX_MESSAGE_PARAMS:
        raise ValueError(f"at most {MAX_MESSAGE_PARAMS} message parameters are allowed")
    for key, item in value.items():
        if not _CODE_PATTERN.fullmatch(key):
            raise ValueError("message parameter names must be snake_case codes")
        if not _PARAM_VALUE_PATTERN.fullmatch(item):
            raise ValueError(
                "message parameter values must be short locale-neutral tokens"
                " (ASCII letters, digits, and . _ : + -), never free text"
            )
    return _freeze_str_map(value)


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
MessageParams = Annotated[
    Mapping[str, str],
    AfterValidator(_freeze_message_params),
    PlainSerializer(_thaw_str_map, return_type=dict[str, str]),
]
"""Read-only, locale-neutral parameters for a catalog message (see `_freeze_message_params`)."""
CurrencyCode = Annotated[str, Field(pattern=r"^[A-Z]{3}$")]


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


class MatchAssumption(StrEnum):
    """Uncertain document data a candidate relies on. Listed so it is never silently trusted."""

    CURRENCY_UNCONFIRMED = "currency_unconfirmed"
    """The document's currency is missing or ambiguous; its total was compared as if it were in
    the account currency. Nothing was converted."""
    DATE_AMBIGUOUS = "date_ambiguous"
    """The document date relied on an assumption: the declared day/month order for a date that
    reads validly both ways (05/03/2026), or a two-digit year read as 20YY."""
    VENDOR_NOT_COMPARABLE = "vendor_not_comparable"
    """Both vendor names exist but share no writing system (e.g. Hebrew receipt, Latin bank
    text), so they could not be compared. Neither a match nor a conflict. A candidate that
    relies on this is never assigned automatically on amount and date alone."""


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
    UPLOAD = "upload"
    """Observed facts about an uploaded file (sanitized filename, upload index, SHA-256)."""


class FindingMessage(StrEnum):
    """Which catalog message renders a finding's title and explanation.

    The catalog keys are ``finding.<value>.title`` and ``finding.<value>.explanation``.
    """

    MISSING_DOCUMENT = "missing_document"
    AMOUNT_MISMATCH = "amount_mismatch"
    DUPLICATE_DOCUMENT_EXACT = "duplicate_document_exact"
    DUPLICATE_DOCUMENT_PROBABLE = "duplicate_document_probable"
    AMBIGUOUS_MATCH = "ambiguous_match"
    INCOMPLETE_DOCUMENT = "incomplete_document"
    UNMATCHED_DOCUMENT = "unmatched_document"

    @property
    def finding_type(self) -> FindingType:
        return _MESSAGE_TYPES[self]


_MESSAGE_TYPES: Final[Mapping[FindingMessage, FindingType]] = MappingProxyType(
    {
        FindingMessage.MISSING_DOCUMENT: FindingType.MISSING_DOCUMENT,
        FindingMessage.AMOUNT_MISMATCH: FindingType.AMOUNT_MISMATCH,
        FindingMessage.DUPLICATE_DOCUMENT_EXACT: FindingType.DUPLICATE_DOCUMENT,
        FindingMessage.DUPLICATE_DOCUMENT_PROBABLE: FindingType.DUPLICATE_DOCUMENT,
        FindingMessage.AMBIGUOUS_MATCH: FindingType.AMBIGUOUS_MATCH,
        FindingMessage.INCOMPLETE_DOCUMENT: FindingType.INCOMPLETE_DOCUMENT,
        FindingMessage.UNMATCHED_DOCUMENT: FindingType.UNMATCHED_DOCUMENT,
    }
)


FINDING_MESSAGE_PARAMS: Final[Mapping[FindingMessage, frozenset[str]]] = MappingProxyType(
    {
        FindingMessage.MISSING_DOCUMENT: frozenset({"amount", "currency", "date", "threshold"}),
        FindingMessage.AMOUNT_MISMATCH: frozenset(
            {"bank_amount", "document_total", "difference", "currency"}
        ),
        FindingMessage.DUPLICATE_DOCUMENT_EXACT: frozenset({"copies"}),
        FindingMessage.DUPLICATE_DOCUMENT_PROBABLE: frozenset({"count"}),
        FindingMessage.AMBIGUOUS_MATCH: frozenset({"count"}),
        FindingMessage.INCOMPLETE_DOCUMENT: frozenset({"count"}),
        FindingMessage.UNMATCHED_DOCUMENT: frozenset(),
    }
)
"""The exact parameter names each finding message takes. Rules must supply exactly these, and
catalog templates may only use these. Amounts are decimal strings in the account currency
(``-12.40``), dates are ISO, counts are integers."""


class FindingHint(StrEnum):
    """Caveats and possible explanations attached to a finding. Catalog key: ``hint.<value>``."""

    TIP_POSSIBLE = "tip_possible"
    """The bank amount is above the document total by a tip-sized amount."""
    LOWER_BANK_AMOUNT = "lower_bank_amount"
    """The bank amount is below the document total (partial payment, discount or credit)."""
    CURRENCY_UNCONFIRMED = "currency_unconfirmed"
    """The document's currency was missing or ambiguous; it was compared as the account's."""
    DATE_AMBIGUOUS = "date_ambiguous"
    """A document date was read using an assumption (declared day/month order, 2-digit year)."""
    VENDOR_NOT_COMPARABLE = "vendor_not_comparable"
    """The vendor names are in different writing systems and could not be compared."""


class NextAction(StrEnum):
    """The suggested next step for the reviewer. Catalog key: ``next_action.<value>``."""

    REQUEST_DOCUMENT = "request_document"
    CONFIRM_AMOUNT_DIFFERENCE = "confirm_amount_difference"
    CONFIRM_DUPLICATE = "confirm_duplicate"
    CHOOSE_MATCHING_DOCUMENT = "choose_matching_document"
    CHECK_DOCUMENT_FIELDS = "check_document_fields"
    UPLOAD_CLEARER_COPY = "upload_clearer_copy"
    CHECK_OTHER_ACCOUNT_OR_PERIOD = "check_other_account_or_period"


class EvidenceField(StrEnum):
    """What a piece of evidence describes. Catalog key: ``evidence.<value>`` (the label)."""

    # Bank export (provenance bank_export)
    TRANSACTION_ID = "transaction.id"
    TRANSACTION_DATE = "transaction.date"
    TRANSACTION_DESCRIPTION = "transaction.description"
    TRANSACTION_AMOUNT = "transaction.amount"
    TRANSACTION_EXTRA_COLUMN = "transaction.extra_column"
    """An extra CSV column. The column name is in `source_ref`, not in the label."""
    # Upload (provenance upload)
    UPLOAD_FILENAME = "upload.filename"
    UPLOAD_SHA256 = "upload.sha256"
    UPLOAD_INDEX = "upload.index"
    # AI extraction (provenance ai_extraction)
    DOCUMENT_TYPE = "document.type"
    DOCUMENT_VENDOR = "document.vendor"
    DOCUMENT_DATE = "document.date"
    DOCUMENT_TOTAL = "document.total"
    DOCUMENT_SUBTOTAL = "document.subtotal"
    DOCUMENT_TAX = "document.tax"
    DOCUMENT_TIP = "document.tip"
    DOCUMENT_CURRENCY = "document.currency"
    DOCUMENT_NUMBER = "document.number"
    # Computed (provenance computed)
    EXTRACTION_STATUS = "computed.extraction_status"
    NORMALIZED_VALUE = "computed.normalized_value"
    """A deterministic reading of an AI-read value; `source_ref` names the field."""
    NORMALIZATION_ISSUE = "computed.normalization_issue"
    EXCLUSION_REASON = "computed.exclusion_reason"
    MATCH_STATUS = "computed.match_status"
    MATCH_SCORE = "computed.match_score"
    VENDOR_SIMILARITY = "computed.vendor_similarity"
    VENDOR_MATCH_KEY = "computed.vendor_match_key"
    AMOUNT_DIFFERENCE = "computed.amount_difference"
    AMOUNT_DIFFERENCE_RATIO = "computed.amount_difference_ratio"
    DATE_DIFFERENCE_DAYS = "computed.date_difference_days"
    ARITHMETIC_DIFFERENCE = "computed.arithmetic_difference"
    THRESHOLD = "computed.threshold"
    CANDIDATE_DOCUMENT = "computed.candidate_document"
    DUPLICATE_KEY = "computed.duplicate_key"


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


class NormalizationIssue(StrEnum):
    """A problem found while normalizing extracted fields. Values are a stable API contract.

    Required fields (vendor, date, total) report `*_missing`. Optional fields (subtotal, tax,
    tip, document number) are often legitimately absent, so they only report `*_invalid`.
    Whether an issue becomes a reviewer-facing finding is decided by the validation rules.
    """

    EXTRACTION_FAILED = "extraction_failed"
    DOCUMENT_UNREADABLE = "document_unreadable"
    DOCUMENT_TYPE_MISSING = "document_type_missing"
    DOCUMENT_TYPE_UNRECOGNIZED = "document_type_unrecognized"
    VENDOR_MISSING = "vendor_missing"
    VENDOR_INVALID = "vendor_invalid"
    DATE_MISSING = "date_missing"
    DATE_INVALID = "date_invalid"
    DATE_AMBIGUOUS = "date_ambiguous"
    TOTAL_MISSING = "total_missing"
    TOTAL_INVALID = "total_invalid"
    TOTAL_NOT_POSITIVE = "total_not_positive"
    SUBTOTAL_INVALID = "subtotal_invalid"
    TAX_INVALID = "tax_invalid"
    TIP_INVALID = "tip_invalid"
    CURRENCY_MISSING = "currency_missing"
    CURRENCY_INVALID = "currency_invalid"
    CURRENCY_AMBIGUOUS = "currency_ambiguous"
    """A symbol shared by several currencies (``$``) with nothing explicit to resolve it, or
    explicit indicators that disagree. The currency is left unset rather than guessed."""
    CURRENCY_UNSUPPORTED = "currency_unsupported"
    """Deprecated. Normalization records facts only; comparing the document currency with the
    account currency is done in matching and validation. To be removed once nothing emits it."""
    DOCUMENT_NUMBER_INVALID = "document_number_invalid"


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
    issues: tuple[NormalizationIssue, ...] = Field(
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
    assumptions: tuple[MatchAssumption, ...] = Field(
        default=(), description="Uncertain document data this candidate's score relies on."
    )

    @model_validator(mode="after")
    def _assumptions_are_unique(self) -> Self:
        if len(set(self.assumptions)) != len(self.assumptions):
            raise ValueError("assumptions must not contain duplicates")
        return self


class MatchResult(DomainModel):
    """Matching outcome for one transaction.

    An unmatched result may still list candidates, kept as evidence for review: documents
    assigned to (or contested between) other transactions with clearly better scores, and
    exact-amount documents from a clearly different merchant.

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
        description=(
            "Other transactions' candidates for this transaction's contested documents, highest"
            " scores first. The matcher may cap this list; each competitor still has its own"
            " ambiguous result."
        ),
    )
    tied_document_ids: tuple[str, ...] = Field(
        default=(),
        description=(
            "The documents that are actually near-tied for this transaction, sorted. Only for"
            " ambiguous results. `candidates` may also list documents that went elsewhere."
        ),
    )

    @model_validator(mode="after")
    def _status_is_consistent(self) -> Self:
        self._check_candidates()
        if self.status is MatchStatus.AMBIGUOUS:
            self._check_ambiguity()
            return self
        if self.ambiguity_reasons or self.competing_candidates or self.tied_document_ids:
            raise ValueError(
                "ambiguity_reasons, competing_candidates and tied_document_ids are only allowed"
                " when ambiguous"
            )
        if self.status is MatchStatus.MATCHED:
            if self.document_id is None or self.document_id not in self._candidate_document_ids:
                raise ValueError("a matched result must reference one of its candidates")
        elif self.document_id is not None:
            raise ValueError("an unmatched result must have no document_id")
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
        self._check_tied_documents()

    def _check_tied_documents(self) -> None:
        # Optional until the matcher fills it; once filled it must be exact.
        tied = self.tied_document_ids
        if not tied:
            return
        if list(tied) != sorted(set(tied)):
            raise ValueError("tied_document_ids must be sorted and unique")
        if not set(tied) <= self._candidate_document_ids:
            raise ValueError("tied_document_ids must reference this result's candidates")
        if AmbiguityReason.MULTIPLE_DOCUMENTS in self.ambiguity_reasons and len(tied) < 2:
            raise ValueError("multiple_documents ambiguity requires at least two tied documents")


# ----------------------------------------------------------------------------- findings


_AI_ONLY_EVIDENCE_FIELDS: Final = ("quote", "page", "model")


class Evidence(DomainModel):
    """One traceable fact supporting a finding.

    Text keeps its original script. `value` is the display form (NFC, bidi controls removed);
    `raw_value` keeps the verbatim cell or extracted value when that differs.
    """

    provenance: Provenance
    field: EvidenceField
    label: NonEmptyStr = Field(
        description="Rendered from the catalog key 'evidence.<field>' in the report locale."
    )
    value: str = Field(description="The value as shown to the reviewer, in its original script.")
    raw_value: str | None = Field(
        default=None, description="Verbatim source value, only when it differs from `value`."
    )
    code: str | None = Field(
        default=None,
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable code when the value is one (e.g. a normalization issue).",
    )
    currency: CurrencyCode | None = Field(
        default=None, description="ISO 4217 code when `value` is a money amount."
    )
    source_ref: NonEmptyStr = Field(description="Locator, e.g. 'csv:row:14' or 'doc:<id>:total'.")
    quote: str | None = Field(
        default=None, description="Verbatim source text the model reported for this value."
    )
    page: int | None = Field(default=None, ge=1, description="1-based page the model reported.")
    model: str | None = Field(default=None, min_length=1, description="Extraction model ID.")

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.provenance is not Provenance.AI_EXTRACTION:
            present = [name for name in _AI_ONLY_EVIDENCE_FIELDS if getattr(self, name) is not None]
            if present:
                raise ValueError(
                    f"{', '.join(present)} are only allowed for ai_extraction evidence"
                )
        if self.raw_value is not None and self.raw_value == self.value:
            raise ValueError("raw_value is only set when it differs from value")
        return self


class RuleRef(DomainModel):
    """The deterministic rule that produced a finding, including the parameters used."""

    rule_id: NonEmptyStr
    rule_version: NonEmptyStr
    # validate_default so the default empty map is frozen too.
    parameters: FrozenStrMap = Field(default_factory=dict, validate_default=True)


class Finding(DomainModel):
    """An exception that needs human review. Never an accounting conclusion.

    The machine-readable contract is `type`, `message`, `params`, `hints`, `next_action` and
    `rule`. `title` and `explanation` are rendered from `message` and `params` in the report
    locale and contain no free text (vendor names and quotes appear only in `evidence`).
    """

    id: NonEmptyStr
    type: FindingType
    severity: Severity
    status: Literal["needs_review"] = "needs_review"
    message: FindingMessage
    # validate_default so the default empty map is frozen too.
    params: MessageParams = Field(default_factory=dict, validate_default=True)
    title: NonEmptyStr
    explanation: NonEmptyStr
    hints: tuple[FindingHint, ...] = ()
    next_action: NextAction
    transaction_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = Field(min_length=1)
    rule: RuleRef

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if not self.transaction_ids and not self.document_ids:
            raise ValueError("a finding must reference at least one transaction or document")
        if self.message.finding_type is not self.type:
            raise ValueError("message must belong to the finding's type")
        expected = FINDING_MESSAGE_PARAMS[self.message]
        if set(self.params) != expected:
            names = ", ".join(sorted(expected)) or "none"
            raise ValueError(f"params for message {self.message.value} must be exactly: {names}")
        if len(set(self.hints)) != len(self.hints):
            raise ValueError("hints must not contain duplicates")
        return self

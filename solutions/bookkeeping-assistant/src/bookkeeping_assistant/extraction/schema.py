"""The extraction contract: what an extractor (LLM or fake) must return for one document.

Extractor output is **untrusted input**. It may come from a model reading attacker-controlled
documents, which could contain prompt-injection text. The schema is therefore strict:
- Every value is a raw *string*, exactly as printed. Numbers, booleans, and objects are rejected,
  so nothing is ever interpreted or computed by the model.
- Unknown keys are rejected. Lengths are capped.
- Every field must be present. A field the document does not show is `null`, never omitted,
  which makes "not found" explicit.

Deterministic interpretation (amounts, dates, vendors, ...) happens in `normalization.py`.
"""

from enum import StrEnum
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from bookkeeping_assistant.models import DomainModel, NonEmptyStr

MAX_VALUE_LENGTH: Final = 200
MAX_SOURCE_TEXT_LENGTH: Final = 500
MAX_PAGE: Final = 1_000

_RawValue = Annotated[str, StringConstraints(max_length=MAX_VALUE_LENGTH)]
_SourceText = Annotated[str, StringConstraints(max_length=MAX_SOURCE_TEXT_LENGTH)]


class _UntrustedModel(BaseModel):
    """Strict, immutable base for extractor output: no coercion, no unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class ExtractedField(_UntrustedModel):
    """One field as read from the document. Every part is model-reported and unverified."""

    value: _RawValue | None = Field(description="The value exactly as printed, or null.")
    source_text: _SourceText | None = Field(
        description="Verbatim quote of the document text the value was read from, or null."
    )
    page: Annotated[int, Field(ge=1, le=MAX_PAGE)] | None = Field(
        description="1-based page number where the value appears, or null."
    )


class DocumentExtraction(_UntrustedModel):
    """Structured output of document extraction. Every key is required; use null if absent."""

    document_type: ExtractedField | None = Field(
        description="What the document is, e.g. 'receipt', 'invoice', 'other', 'unreadable'."
    )
    vendor_name: ExtractedField | None
    document_date: ExtractedField | None
    total: ExtractedField | None
    subtotal: ExtractedField | None
    tax: ExtractedField | None
    tip: ExtractedField | None
    currency: ExtractedField | None
    document_number: ExtractedField | None = Field(
        description="Receipt, invoice, or order number as printed."
    )


class ExtractionStatus(StrEnum):
    OK = "ok"
    FAILED = "failed"


class ExtractionErrorCode(StrEnum):
    """Why an extraction failed. Phase 6 adds provider-specific codes (refusal, timeout, ...)."""

    INVALID_OUTPUT = "invalid_output"
    """The extractor returned output that does not match `DocumentExtraction`."""
    NO_FIXTURE = "no_fixture"
    """`FakeExtractor` has no fixture for this document."""


class ExtractionResult(DomainModel):
    """The outcome of extracting one document. Never an exception for per-document failures."""

    document_id: NonEmptyStr
    status: ExtractionStatus
    extraction: DocumentExtraction | None = None
    error_code: ExtractionErrorCode | None = None
    model: str | None = Field(default=None, description="Extractor or model that produced this.")

    @model_validator(mode="after")
    def _status_is_consistent(self) -> Self:
        if self.status is ExtractionStatus.OK:
            if self.extraction is None or self.error_code is not None:
                raise ValueError("an ok result needs an extraction and no error_code")
        elif self.extraction is not None or self.error_code is None:
            raise ValueError("a failed result needs an error_code and no extraction")
        return self

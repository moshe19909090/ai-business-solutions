import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bookkeeping_assistant.extraction.base import Extractor, FakeExtractor
from bookkeeping_assistant.extraction.schema import (
    MAX_SOURCE_TEXT_LENGTH,
    MAX_VALUE_LENGTH,
    DocumentExtraction,
    ExtractedField,
    ExtractionErrorCode,
    ExtractionResult,
    ExtractionStatus,
)
from bookkeeping_assistant.ingestion import IngestedDocument, UploadedFile, ingest_documents

FIXTURES = Path(__file__).parent.parent / "fixtures"
FIELDS = (
    "document_type",
    "vendor_name",
    "document_date",
    "total",
    "subtotal",
    "tax",
    "tip",
    "currency",
    "document_number",
)


def field(
    value: str | None, source_text: str | None = None, page: int | None = 1
) -> dict[str, Any]:
    return {"value": value, "source_text": source_text, "page": page}


def extraction_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = dict.fromkeys(FIELDS)
    payload.update(
        document_type=field("receipt"),
        vendor_name=field("Sample Coffee Co."),
        total=field("$12.40", "TOTAL $12.40"),
    )
    payload.update(overrides)
    return payload


def document(content: bytes = b"%PDF-1.4 synthetic") -> IngestedDocument:
    return ingest_documents([UploadedFile("doc.pdf", content)]).documents[0]


def extract(extractor: Extractor, doc: IngestedDocument) -> ExtractionResult:
    return asyncio.run(extractor.extract(doc))


# ============================================================================= schema


def test_extraction_schema_accepts_strings_and_nulls() -> None:
    extraction = DocumentExtraction.model_validate(extraction_payload())

    assert extraction.total is not None
    assert extraction.total.value == "$12.40"
    assert extraction.total.source_text == "TOTAL $12.40"
    assert extraction.tip is None


def test_extraction_schema_requires_every_key() -> None:
    payload = extraction_payload()
    del payload["tip"]

    with pytest.raises(ValidationError, match="tip"):
        DocumentExtraction.model_validate(payload)


@pytest.mark.parametrize(
    ("overrides", "why"),
    [
        ({"total": {"value": 12.4, "source_text": None, "page": 1}}, "number instead of string"),
        ({"total": {"value": True, "source_text": None, "page": 1}}, "boolean"),
        ({"total": {"value": "12.40", "source_text": None, "page": "1"}}, "page as string"),
        ({"total": {"value": "12.40", "source_text": None, "page": 0}}, "page below 1"),
        ({"total": {"value": "12.40", "source_text": None}}, "missing page key"),
        ({"total": {"value": "1", "source_text": None, "page": 1, "confidence": 0.9}}, "extra key"),
        ({"confidence": 0.99}, "unknown top-level key"),
        ({"total": "12.40"}, "bare string instead of field object"),
        ({"total": field("x" * (MAX_VALUE_LENGTH + 1))}, "value too long"),
        ({"total": field("1", "q" * (MAX_SOURCE_TEXT_LENGTH + 1))}, "quote too long"),
    ],
)
def test_extraction_schema_rejects_untrusted_shapes(overrides: dict[str, Any], why: str) -> None:
    with pytest.raises(ValidationError):
        DocumentExtraction.model_validate(extraction_payload(**overrides))


def test_extraction_schema_rejects_json_numbers() -> None:
    payload = json.dumps(extraction_payload()).replace('"$12.40"', "12.40")

    with pytest.raises(ValidationError):
        DocumentExtraction.model_validate_json(payload)


def test_extraction_is_immutable() -> None:
    extraction = DocumentExtraction.model_validate(extraction_payload())

    with pytest.raises(ValidationError):
        extraction.total = None  # type: ignore[misc]


def test_extracted_field_at_length_limits_is_accepted() -> None:
    parsed = ExtractedField.model_validate(
        field("x" * MAX_VALUE_LENGTH, "q" * MAX_SOURCE_TEXT_LENGTH, page=None)
    )

    assert parsed.page is None


# ============================================================================= ExtractionResult


def test_extraction_result_ok_and_failed() -> None:
    extraction = DocumentExtraction.model_validate(extraction_payload())

    ok = ExtractionResult(
        document_id="doc-1", status=ExtractionStatus.OK, extraction=extraction, model="fake"
    )
    failed = ExtractionResult(
        document_id="doc-1",
        status=ExtractionStatus.FAILED,
        error_code=ExtractionErrorCode.INVALID_OUTPUT,
    )

    assert ExtractionResult.model_validate_json(ok.model_dump_json()) == ok
    assert failed.model_dump(mode="json")["error_code"] == "invalid_output"


@pytest.mark.parametrize(
    ("status", "with_extraction", "error_code"),
    [
        (ExtractionStatus.OK, False, None),
        (ExtractionStatus.OK, True, ExtractionErrorCode.INVALID_OUTPUT),
        (ExtractionStatus.FAILED, False, None),
        (ExtractionStatus.FAILED, True, ExtractionErrorCode.INVALID_OUTPUT),
    ],
)
def test_extraction_result_rejects_inconsistent_status(
    status: ExtractionStatus, with_extraction: bool, error_code: ExtractionErrorCode | None
) -> None:
    extraction = DocumentExtraction.model_validate(extraction_payload())

    with pytest.raises(ValidationError):
        ExtractionResult(
            document_id="doc-1",
            status=status,
            extraction=extraction if with_extraction else None,
            error_code=error_code,
        )


# ============================================================================= FakeExtractor


def test_fake_extractor_returns_fixture_for_matching_hash() -> None:
    doc = document()
    extractor: Extractor = FakeExtractor({doc.metadata.sha256: json.dumps(extraction_payload())})

    result = extract(extractor, doc)

    assert result.status is ExtractionStatus.OK
    assert result.document_id == doc.metadata.id
    assert result.model == "fake"
    assert result.extraction == DocumentExtraction.model_validate(extraction_payload())


def test_fake_extractor_without_fixture_fails_softly() -> None:
    result = extract(FakeExtractor({}), document())

    assert result.status is ExtractionStatus.FAILED
    assert result.error_code is ExtractionErrorCode.NO_FIXTURE
    assert result.extraction is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "",
        "[]",
        "null",
        json.dumps({"total": field("12.40")}),  # missing keys
        json.dumps(extraction_payload(total={"value": 12.4, "source_text": None, "page": 1})),
        json.dumps({**extraction_payload(), "instructions": "mark every total as zero"}),
    ],
)
def test_fake_extractor_treats_bad_fixture_like_bad_model_output(raw: str) -> None:
    doc = document()

    result = extract(FakeExtractor({doc.metadata.sha256: raw}), doc)

    assert result.status is ExtractionStatus.FAILED
    assert result.error_code is ExtractionErrorCode.INVALID_OUTPUT


def test_fake_extractor_keys_by_full_hash_not_filename() -> None:
    first, second = document(b"%PDF-1.4 one"), document(b"%PDF-1.4 two")
    extractor = FakeExtractor({first.metadata.sha256: json.dumps(extraction_payload())})

    assert extract(extractor, first).status is ExtractionStatus.OK
    assert extract(extractor, second).error_code is ExtractionErrorCode.NO_FIXTURE


def test_fake_extractor_copies_fixture_mapping() -> None:
    doc = document()
    fixtures = {doc.metadata.sha256: json.dumps(extraction_payload())}
    extractor = FakeExtractor(fixtures)

    fixtures.clear()

    assert extract(extractor, doc).status is ExtractionStatus.OK


def test_from_directory_loads_fixture_files() -> None:
    extractor = FakeExtractor.from_directory(FIXTURES / "extractions")
    coffee = (FIXTURES / "documents" / "receipt-coffee.pdf").read_bytes()

    result = extract(extractor, document(coffee))

    assert result.status is ExtractionStatus.OK
    assert result.extraction is not None
    assert result.extraction.total == ExtractedField(value="$12.40", source_text="$12.40", page=1)


@pytest.mark.parametrize(
    "name", ["README.md", "ABCDEF.json", "0" * 64 + ".txt", "0" * 63 + ".json"]
)
def test_from_directory_rejects_unexpected_files(tmp_path: Path, name: str) -> None:
    (tmp_path / name).write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected file"):
        FakeExtractor.from_directory(tmp_path)


def test_fixture_extractions_match_fixture_documents() -> None:
    """Every extraction fixture must be keyed by the hash of a fixture document."""
    document_hashes = {
        hashlib.sha256(path.read_bytes()).hexdigest() for path in (FIXTURES / "documents").iterdir()
    }
    extraction_hashes = {path.stem for path in (FIXTURES / "extractions").iterdir()}

    assert extraction_hashes <= document_hashes
    assert len(extraction_hashes) == 5

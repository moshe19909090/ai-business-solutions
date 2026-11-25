"""Stage 1: ingestion. Validate uploads and parse the canonical transactions CSV.

Pure functions over in-memory bytes: no file system, network, or logging.

Canonical CSV (UTF-8, optional BOM, header row required):

    date,description,amount[,transaction_id]

- Header names are case-insensitive and trimmed. Extra columns are allowed. They are kept in
  `BankTransaction.raw` but otherwise ignored.
- ``date``: ``YYYY-MM-DD`` or ``M/D/YYYY`` (US month-first).
- ``amount``: US format, negative means outflow (see `money.parse_amount`).
- ``transaction_id`` (optional): used when present and non-blank. Otherwise an ID is derived
  from the row's line number, so the same file always yields the same IDs.

Error messages identify the row and column but never echo cell values. That keeps them safe
to log or display without leaking financial data.
"""

import csv
import hashlib
import io
import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, ClassVar, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from bookkeeping_assistant.models import (
    DOCUMENT_ID_LENGTH,
    BankTransaction,
    DomainModel,
    MediaType,
    NonEmptyStr,
    SourceDocument,
)
from bookkeeping_assistant.money import AmountParseError, parse_amount

MEBIBYTE: Final = 1024 * 1024

REQUIRED_COLUMNS: Final = ("date", "description", "amount")
TRANSACTION_ID_COLUMN: Final = "transaction_id"
MAX_REPORTED_ISSUES: Final = 100
MAX_FILENAME_LENGTH: Final = 120
MAX_TRANSACTION_ID_LENGTH: Final = 64

_TRANSACTION_ID_PATTERN: Final = re.compile(
    rf"[A-Za-z0-9][A-Za-z0-9._:\-]{{0,{MAX_TRANSACTION_ID_LENGTH - 1}}}", re.ASCII
)
_ISO_DATE: Final = re.compile(r"([0-9]{4})-([0-9]{2})-([0-9]{2})", re.ASCII)
_US_DATE: Final = re.compile(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})", re.ASCII)
_UNSAFE_FILENAME_CHARS: Final = re.compile(r"[^A-Za-z0-9._() \-]+", re.ASCII)

# File signatures ("magic bytes"). The signature decides the type, never the extension.
_SIGNATURES: Final[tuple[tuple[bytes, MediaType], ...]] = (
    (b"%PDF-", MediaType.PDF),
    (b"\x89PNG\r\n\x1a\n", MediaType.PNG),
    (b"\xff\xd8\xff", MediaType.JPEG),
)
_EXTENSIONS: Final = {MediaType.PDF: "pdf", MediaType.PNG: "png", MediaType.JPEG: "jpg"}


# ----------------------------------------------------------------------------- limits


class IngestionLimits(BaseModel):
    """Upper bounds on upload sizes. The defaults follow the approved MVP plan."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    max_csv_bytes: Annotated[int, Field(ge=1)] = 5 * MEBIBYTE
    max_csv_rows: Annotated[int, Field(ge=1)] = 2_000
    max_csv_columns: Annotated[int, Field(ge=len(REQUIRED_COLUMNS))] = 100
    max_documents: Annotated[int, Field(ge=1)] = 25
    max_document_bytes: Annotated[int, Field(ge=1)] = 10 * MEBIBYTE


DEFAULT_LIMITS: Final = IngestionLimits()


# ----------------------------------------------------------------------------- errors


class CsvIssue(DomainModel):
    """One problem in the transactions CSV. `row_number` is None for file-level problems."""

    row_number: int | None = Field(default=None, ge=1)
    column: str | None = None
    message: NonEmptyStr


class DocumentIssueCode(StrEnum):
    EMPTY_FILE = "empty_file"
    FILE_TOO_LARGE = "file_too_large"
    UNSUPPORTED_TYPE = "unsupported_type"
    NO_DOCUMENTS = "no_documents"


class DocumentIssue(DomainModel):
    """One problem with an uploaded document. `upload_index` is 0-based; None means file-level."""

    upload_index: int | None = Field(default=None, ge=0)
    filename: str | None = Field(default=None, description="Sanitized name, if available.")
    code: DocumentIssueCode
    message: NonEmptyStr


class IngestionError(Exception):
    """Base class for all ingestion failures. `code` is a stable, machine-readable identifier."""

    code: ClassVar[str] = "ingestion_error"


class CsvValidationError(IngestionError):
    """The CSV is malformed. `issues` lists every problem found (capped)."""

    code = "invalid_csv"

    def __init__(self, issues: Sequence[CsvIssue]) -> None:
        self.issues: tuple[CsvIssue, ...] = tuple(issues)
        super().__init__(f"transactions CSV has {len(self.issues)} issue(s)")


class DocumentValidationError(IngestionError):
    """One or more documents were rejected. `issues` lists every rejected upload."""

    code = "invalid_documents"

    def __init__(self, issues: Sequence[DocumentIssue]) -> None:
        self.issues: tuple[DocumentIssue, ...] = tuple(issues)
        super().__init__(f"{len(self.issues)} document issue(s)")


class LimitExceededError(IngestionError):
    """A request-level size or count limit was exceeded."""

    code = "limit_exceeded"

    def __init__(self, limit: str, maximum: int) -> None:
        self.limit = limit
        self.maximum = maximum
        super().__init__(f"{limit} exceeds the maximum of {maximum}")


# ----------------------------------------------------------------------------- results


class ParsedTransactions(DomainModel):
    """The validated transactions CSV."""

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    transactions: tuple[BankTransaction, ...] = Field(min_length=1)


@dataclass(frozen=True, slots=True)
class UploadedFile:
    """A raw upload as received. `filename` is untrusted and may be missing."""

    filename: str | None
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class IngestedDocument:
    """A validated document: metadata plus the bytes needed for extraction.

    A dataclass rather than a Pydantic model, so the content is never serialized into a report
    by accident. It is also left out of `repr`, so it cannot leak into logs.
    """

    metadata: SourceDocument
    content: bytes = field(repr=False)


class DuplicateUpload(DomainModel):
    """A byte-identical copy of an earlier upload. Feeds the exact-duplicate finding (Phase 5)."""

    document_id: NonEmptyStr = Field(description="ID of the first upload with this content.")
    upload_index: int = Field(ge=0, description="0-based position of the duplicate upload.")
    filename: NonEmptyStr = Field(description="Sanitized name of the duplicate upload.")


@dataclass(frozen=True, slots=True)
class IngestedDocuments:
    """Unique documents in upload order, plus any byte-identical duplicates that were folded."""

    documents: tuple[IngestedDocument, ...]
    duplicates: tuple[DuplicateUpload, ...]


# ----------------------------------------------------------------------------- CSV


def parse_csv_date(value: str) -> date:
    """Parse ``YYYY-MM-DD`` or ``M/D/YYYY`` (US month-first). Raises `ValueError` otherwise."""
    text = value.strip()
    if match := _ISO_DATE.fullmatch(text):
        year, month, day = (int(part) for part in match.groups())
    elif match := _US_DATE.fullmatch(text):
        month, day, year = (int(part) for part in match.groups())
    else:
        raise ValueError("invalid date: expected YYYY-MM-DD or MM/DD/YYYY")
    try:
        return date(year, month, day)
    except ValueError as exc:
        raise ValueError("invalid date: not a real calendar date") from exc


def parse_transactions_csv(
    data: bytes, *, limits: IngestionLimits = DEFAULT_LIMITS
) -> ParsedTransactions:
    """Parse and validate the canonical transactions CSV.

    Raises:
        LimitExceededError: the file is over `max_csv_bytes` or has more than `max_csv_rows`
            data rows.
        CsvValidationError: anything else is wrong, including a header with more than
            `max_csv_columns` columns. Every problem is reported (up to `MAX_REPORTED_ISSUES`,
            then a truncation note), not just the first. Issues collected before an
            unrecoverable syntax error are kept, and the syntax error is appended.
    """
    if len(data) > limits.max_csv_bytes:
        raise LimitExceededError("csv_bytes", limits.max_csv_bytes)
    records = _records(_decode_csv(data))

    header_line, header = next(records, (0, []))
    if not header:
        raise CsvValidationError([CsvIssue(message="the file is empty")])
    columns = _validate_header(header, header_line, limits)
    transactions, issues = _parse_rows(records, header, columns, limits)

    if not issues and not transactions:
        issues.append(CsvIssue(message="the file contains no transactions"))
    if issues:
        raise CsvValidationError(issues)
    return ParsedTransactions(
        sha256=hashlib.sha256(data).hexdigest(), transactions=tuple(transactions)
    )


def _decode_csv(data: bytes) -> str:
    if not data.strip():
        raise CsvValidationError([CsvIssue(message="the file is empty")])
    if b"\x00" in data:
        raise CsvValidationError([CsvIssue(message="the file contains NUL bytes (binary file?)")])
    try:
        return data.decode("utf-8-sig")  # utf-8-sig strips a leading BOM if present
    except UnicodeDecodeError as exc:
        raise CsvValidationError([CsvIssue(message="the file must be UTF-8 encoded")]) from exc


def _records(text: str) -> Iterator[tuple[int, list[str]]]:
    """Yield (starting line number, cells) for each non-blank CSV record.

    A quoted field can span lines, so a record starts on the line after the previous one ended.
    """
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    previous_end = 0
    try:
        for row in reader:
            start, previous_end = previous_end + 1, reader.line_num
            if any(cell.strip() for cell in row):
                yield start, row
    except csv.Error as exc:
        issue = CsvIssue(row_number=previous_end + 1, message=f"malformed CSV: {exc}")
        raise CsvValidationError([issue]) from exc


def _validate_header(header: list[str], line: int, limits: IngestionLimits) -> dict[str, int]:
    """Map normalized column names to indexes, or raise with the header problems (capped)."""
    if len(header) > limits.max_csv_columns:
        message = f"the header has {len(header)} columns; at most {limits.max_csv_columns} allowed"
        raise CsvValidationError([CsvIssue(row_number=line, message=message)])

    columns: dict[str, int] = {}
    issues: list[CsvIssue] = []
    truncated = False

    def problem(message: str, column: str | None = None) -> None:
        nonlocal truncated
        if len(issues) >= MAX_REPORTED_ISSUES:
            truncated = True  # count it, but do not build the issue object
            return
        issues.append(CsvIssue(row_number=line, column=column, message=message))

    for index, name in enumerate(header):
        normalized = name.strip().lower()
        if not normalized:
            problem(f"column {index + 1} has no name")
        elif normalized in columns:
            problem("duplicate column name", normalized)
        else:
            columns[normalized] = index
    for required in REQUIRED_COLUMNS:
        if required not in columns:
            problem("required column is missing", required)
    if issues:
        raise CsvValidationError(_capped(issues, truncated=truncated))
    return columns


def _parse_rows(
    records: Iterator[tuple[int, list[str]]],
    header: list[str],
    columns: dict[str, int],
    limits: IngestionLimits,
) -> tuple[list[BankTransaction], list[CsvIssue]]:
    transactions: list[BankTransaction] = []
    issues: list[CsvIssue] = []
    seen_ids: dict[str, int] = {}
    truncated = False

    try:
        for data_rows, (row_number, row) in enumerate(records, start=1):
            if data_rows > limits.max_csv_rows:
                raise LimitExceededError("csv_rows", limits.max_csv_rows)
            if len(issues) >= MAX_REPORTED_ISSUES:
                truncated = True
                continue  # keep reading so the row limit is still enforced

            row_issues, transaction = _parse_row(row, row_number, header, columns)
            if transaction is not None:
                first_seen = seen_ids.setdefault(transaction.id, row_number)
                if first_seen == row_number:
                    transactions.append(transaction)
                else:
                    row_issues.append(
                        CsvIssue(
                            row_number=row_number,
                            column=TRANSACTION_ID_COLUMN,
                            message=f"duplicate transaction ID (first used on line {first_seen})",
                        )
                    )
            issues.extend(row_issues)
    except CsvValidationError as syntax_error:
        # Unrecoverable malformed CSV: stop reading, but keep the issues found so far.
        issues.extend(syntax_error.issues)

    return transactions, _capped(issues, truncated=truncated)


def _capped(issues: list[CsvIssue], *, truncated: bool) -> list[CsvIssue]:
    """Trim `issues` to `MAX_REPORTED_ISSUES` and add a note if anything was left out."""
    if truncated or len(issues) > MAX_REPORTED_ISSUES:
        del issues[MAX_REPORTED_ISSUES:]
        issues.append(CsvIssue(message=f"stopped reporting after {MAX_REPORTED_ISSUES} issues"))
    return issues


def _parse_row(
    row: list[str], row_number: int, header: list[str], columns: dict[str, int]
) -> tuple[list[CsvIssue], BankTransaction | None]:
    if len(row) != len(header):
        message = f"expected {len(header)} fields, found {len(row)}"
        return [CsvIssue(row_number=row_number, message=message)], None

    def cell(name: str) -> str:
        return row[columns[name]].strip()

    issues: list[CsvIssue] = []

    def problem(column: str, message: str) -> None:
        issues.append(CsvIssue(row_number=row_number, column=column, message=message))

    posted_date: date | None = None
    try:
        posted_date = parse_csv_date(cell("date"))
    except ValueError as exc:
        problem("date", str(exc))

    description = cell("description")
    if not description:
        problem("description", "description is empty")

    amount: Decimal | None = None
    try:
        amount = parse_amount(cell("amount"))
    except AmountParseError:
        problem("amount", "invalid amount: expected a US amount such as -1,234.56 or (12.00)")

    transaction_id = f"txn-{row_number:06d}"
    if TRANSACTION_ID_COLUMN in columns and (provided := cell(TRANSACTION_ID_COLUMN)):
        if _TRANSACTION_ID_PATTERN.fullmatch(provided):
            transaction_id = provided
        else:
            problem(
                TRANSACTION_ID_COLUMN,
                f"transaction ID must be 1-{MAX_TRANSACTION_ID_LENGTH} characters: "
                "letters, digits, '.', '_', ':' or '-'",
            )

    if issues or posted_date is None or amount is None:
        return issues, None
    try:
        transaction = BankTransaction(
            id=transaction_id,
            row_number=row_number,
            posted_date=posted_date,
            description=description,
            amount=amount,
            raw=dict(zip(header, row, strict=True)),
        )
    except ValidationError:  # defensive: inputs above are already validated
        return [CsvIssue(row_number=row_number, message="row could not be validated")], None
    return issues, transaction


# ----------------------------------------------------------------------------- documents


def detect_media_type(content: bytes) -> MediaType | None:
    """Identify PDF, PNG, or JPEG from the file signature. Returns None for anything else."""
    for signature, media_type in _SIGNATURES:
        if content.startswith(signature):
            return media_type
    return None


def sanitize_filename(name: str | None, *, fallback: str) -> str:
    """Produce a safe display name. The result is never used as a file system path.

    Keeps only the last path component. Folds accents to ASCII and replaces anything outside
    ``A-Z a-z 0-9 . _ ( ) - space`` with ``_``. Strips leading dots and caps the length while
    keeping the extension. Returns `fallback` if nothing meaningful remains.
    """
    if not name:
        return fallback
    base = re.split(r"[\\/]", name)[-1]
    ascii_text = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", ascii_text)
    cleaned = re.sub(r"_+", "_", re.sub(r" +", " ", cleaned)).strip()
    stem = cleaned.rpartition(".")[0] if "." in cleaned else cleaned
    if not stem.strip("._-() "):  # nothing meaningful left, e.g. "..", "___", "領収書.pdf"
        return fallback
    cleaned = cleaned.lstrip(". ")
    if len(cleaned) > MAX_FILENAME_LENGTH:
        stem, dot, extension = cleaned.rpartition(".")
        if dot and stem and len(extension) <= 10:
            cleaned = f"{stem[: MAX_FILENAME_LENGTH - len(extension) - 1]}.{extension}"
        else:
            cleaned = cleaned[:MAX_FILENAME_LENGTH]
    return cleaned


def ingest_documents(
    uploads: Sequence[UploadedFile], *, limits: IngestionLimits = DEFAULT_LIMITS
) -> IngestedDocuments:
    """Validate uploaded documents, hash them, and fold byte-identical duplicates.

    Raises:
        LimitExceededError: more than `max_documents` uploads.
        DocumentValidationError: no uploads, or any upload is empty, too large, or not a
            PDF, PNG, or JPEG by signature. Every rejected upload is reported.
    """
    if len(uploads) > limits.max_documents:
        raise LimitExceededError("document_count", limits.max_documents)
    if not uploads:
        raise DocumentValidationError(
            [
                DocumentIssue(
                    code=DocumentIssueCode.NO_DOCUMENTS, message="at least one document is required"
                )
            ]
        )

    issues: list[DocumentIssue] = []
    accepted: list[tuple[int, bytes, MediaType, str]] = []
    for index, upload in enumerate(uploads):
        checked = _check_upload(upload.content, limits)
        fallback = f"document-{index + 1}"
        if isinstance(checked, MediaType):
            fallback = f"{fallback}.{_EXTENSIONS[checked]}"
        filename = sanitize_filename(upload.filename, fallback=fallback)
        if isinstance(checked, MediaType):
            accepted.append((index, upload.content, checked, filename))
        else:
            code, message = checked
            issues.append(
                DocumentIssue(upload_index=index, filename=filename, code=code, message=message)
            )
    if issues:
        raise DocumentValidationError(issues)

    documents: dict[str, IngestedDocument] = {}
    duplicates: list[DuplicateUpload] = []
    for index, content, media_type, filename in accepted:
        digest = _sha256_hex(content)
        document_id = digest[:DOCUMENT_ID_LENGTH]
        existing = documents.get(document_id)
        if existing is not None:
            if existing.metadata.sha256 != digest:  # 64-bit prefix collision: astronomically rare
                raise IngestionError("document ID collision between different files")
            duplicates.append(
                DuplicateUpload(document_id=document_id, upload_index=index, filename=filename)
            )
            continue
        metadata = SourceDocument(
            id=document_id,
            filename=filename,
            media_type=media_type,
            size_bytes=len(content),
            sha256=digest,
        )
        documents[document_id] = IngestedDocument(metadata=metadata, content=content)

    return IngestedDocuments(documents=tuple(documents.values()), duplicates=tuple(duplicates))


def _check_upload(
    content: bytes, limits: IngestionLimits
) -> MediaType | tuple[DocumentIssueCode, str]:
    """Return the detected media type, or an (issue code, message) pair."""
    if not content:
        return DocumentIssueCode.EMPTY_FILE, "the file is empty"
    if len(content) > limits.max_document_bytes:
        return (
            DocumentIssueCode.FILE_TOO_LARGE,
            f"the file exceeds the maximum of {limits.max_document_bytes} bytes",
        )
    media_type = detect_media_type(content)
    if media_type is None:
        return DocumentIssueCode.UNSUPPORTED_TYPE, "only PDF, PNG, and JPEG files are supported"
    return media_type


def _sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()

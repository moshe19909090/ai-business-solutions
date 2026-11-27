import hashlib
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from bookkeeping_assistant.ingestion import (
    MAX_FILENAME_LENGTH,
    MAX_REPORTED_ISSUES,
    CsvIssue,
    CsvValidationError,
    DocumentIssueCode,
    DocumentValidationError,
    IngestionError,
    IngestionLimits,
    LimitExceededError,
    UploadedFile,
    detect_media_type,
    ingest_documents,
    parse_csv_date,
    parse_transactions_csv,
    sanitize_filename,
)
from bookkeeping_assistant.models import Direction, MediaType
from bookkeeping_assistant.profiles import CsvEncoding, DateOrder

FIXTURES = Path(__file__).parent.parent / "fixtures"

PDF = b"%PDF-1.7\n%synthetic\n1 0 obj\n<<>>\nendobj\n%%EOF\n"
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00" + b"\x00" * 16 + b"\xff\xd9"

HEADER = "date,description,amount\n"


def csv_bytes(text: str) -> bytes:
    return text.encode("utf-8")


def csv_issues(text: str | bytes, **kwargs: object) -> tuple[CsvIssue, ...]:
    data = text if isinstance(text, bytes) else csv_bytes(text)
    with pytest.raises(CsvValidationError) as excinfo:
        parse_transactions_csv(data, **kwargs)  # type: ignore[arg-type]
    return excinfo.value.issues


def issue_keys(issues: tuple[CsvIssue, ...]) -> list[tuple[int | None, str | None]]:
    return [(issue.row_number, issue.column) for issue in issues]


# ============================================================================= CSV: happy paths


def test_parses_fixture_file() -> None:
    data = (FIXTURES / "transactions_valid.csv").read_bytes()

    parsed = parse_transactions_csv(data)

    assert parsed.sha256 == hashlib.sha256(data).hexdigest()
    assert [t.id for t in parsed.transactions] == [
        "BANK-1001",
        "BANK-1002",
        "BANK-1003",
        "txn-000005",  # blank transaction_id -> derived from line number
        "BANK-1005",
    ]
    assert [t.row_number for t in parsed.transactions] == [2, 3, 4, 5, 6]
    assert [t.posted_date for t in parsed.transactions] == [
        date(2026, 3, 2),
        date(2026, 3, 3),
        date(2026, 3, 5),
        date(2026, 3, 6),
        date(2026, 3, 7),
    ]
    assert [str(t.amount) for t in parsed.transactions] == [
        "-12.40",
        "-245.99",
        "1500.00",
        "-49.00",
        "0.00",
    ]
    assert [t.direction for t in parsed.transactions] == [
        Direction.OUTFLOW,
        Direction.OUTFLOW,
        Direction.INFLOW,
        Direction.OUTFLOW,
        Direction.NEUTRAL,
    ]
    second = parsed.transactions[1]
    assert second.description == "OFFICE SUPPLY DEPOT #123, ONLINE"
    assert dict(second.raw) == {
        "date": "03/03/2026",
        "description": "OFFICE SUPPLY DEPOT #123, ONLINE",
        "amount": "-245.99",
        "transaction_id": "BANK-1002",
    }


def test_parsing_is_deterministic() -> None:
    data = (FIXTURES / "transactions_valid.csv").read_bytes()

    assert parse_transactions_csv(data) == parse_transactions_csv(data)


def test_generated_ids_without_transaction_id_column() -> None:
    parsed = parse_transactions_csv(csv_bytes(HEADER + "2026-03-02,A,-1.00\n2026-03-03,B,-2.00\n"))

    assert [t.id for t in parsed.transactions] == ["txn-000002", "txn-000003"]


def test_handles_utf8_bom_and_crlf() -> None:
    data = b"\xef\xbb\xbf" + b"date,description,amount\r\n2026-03-02,COFFEE,-12.40\r\n"

    parsed = parse_transactions_csv(data)

    assert parsed.transactions[0].description == "COFFEE"
    assert "date" in parsed.transactions[0].raw  # BOM not glued to the first header


def test_header_is_case_insensitive_and_trimmed_and_columns_any_order() -> None:
    parsed = parse_transactions_csv(csv_bytes(" Amount ,DATE, Description\n-5.00,2026-03-02,X\n"))

    transaction = parsed.transactions[0]
    assert (transaction.amount, transaction.posted_date) == (Decimal("-5.00"), date(2026, 3, 2))
    assert dict(transaction.raw) == {" Amount ": "-5.00", "DATE": "2026-03-02", " Description": "X"}


def test_cell_whitespace_is_trimmed_for_parsing_but_raw_is_verbatim() -> None:
    parsed = parse_transactions_csv(csv_bytes(HEADER + "  2026-03-02 ,  COFFEE  ,  -12.40  \n"))

    transaction = parsed.transactions[0]
    assert transaction.description == "COFFEE"
    assert transaction.amount == Decimal("-12.40")
    assert transaction.raw["description"] == "  COFFEE  "


def test_extra_columns_are_kept_in_raw() -> None:
    parsed = parse_transactions_csv(
        csv_bytes("date,description,amount,memo\n2026-03-02,COFFEE,-12.40,team meeting\n")
    )

    assert parsed.transactions[0].raw["memo"] == "team meeting"


def test_blank_lines_are_skipped_and_line_numbers_stay_accurate() -> None:
    text = "\n" + HEADER + "\n2026-03-02,A,-1.00\n , , \n\n2026-03-03,B,-2.00\n\n\n"

    parsed = parse_transactions_csv(csv_bytes(text))

    assert [t.row_number for t in parsed.transactions] == [4, 7]


def test_multiline_quoted_field_reports_starting_line() -> None:
    text = HEADER + '2026-03-02,"LINE ONE\nLINE TWO",-1.00\n2026-03-03,B,-2.00\n'

    parsed = parse_transactions_csv(csv_bytes(text))

    assert parsed.transactions[0].description == "LINE ONE\nLINE TWO"
    assert [t.row_number for t in parsed.transactions] == [2, 4]


def test_no_trailing_newline() -> None:
    parsed = parse_transactions_csv(csv_bytes(HEADER + "2026-03-02,A,-1.00"))

    assert len(parsed.transactions) == 1


def test_exactly_at_limits_is_accepted() -> None:
    text = HEADER + "2026-03-02,A,-1.00\n" * 3
    data = csv_bytes(text)
    limits = IngestionLimits(max_csv_rows=3, max_csv_bytes=len(data))

    assert len(parse_transactions_csv(data, limits=limits).transactions) == 3


# ============================================================================= CSV: dates


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-03-04", date(2026, 3, 4)),
        ("03/04/2026", date(2026, 3, 4)),  # US: month first
        ("3/4/2026", date(2026, 3, 4)),
        ("12/31/2026", date(2026, 12, 31)),
        ("2/29/2028", date(2028, 2, 29)),  # leap year
        ("  2026-03-04  ", date(2026, 3, 4)),
    ],
)
def test_parse_csv_date_accepts(value: str, expected: date) -> None:
    assert parse_csv_date(value) == expected


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("", "expected YYYY-MM-DD"),
        ("2026/03/04", "expected YYYY-MM-DD"),
        ("03-04-2026", "expected YYYY-MM-DD"),
        ("04.03.2026", "expected YYYY-MM-DD"),
        ("2026-3-4", "expected YYYY-MM-DD"),
        ("3/4/26", "expected YYYY-MM-DD"),  # two-digit years are ambiguous
        ("March 4, 2026", "expected YYYY-MM-DD"),
        ("2026-03-04T10:00:00", "expected YYYY-MM-DD"),
        ("\uff12\uff10\uff12\uff16-03-04", "expected YYYY-MM-DD"),  # full-width digits
        ("2026-02-30", "not a real calendar date"),
        ("2/29/2027", "not a real calendar date"),
        ("13/01/2026", "not a real calendar date"),  # DD/MM is not accepted
        ("0000-01-01", "not a real calendar date"),
    ],
)
def test_parse_csv_date_rejects(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_csv_date(value)


# ============================================================================= CSV: row errors


def test_collects_all_row_errors_with_row_numbers_and_columns() -> None:
    text = (
        HEADER
        + "2026-02-30,A,-1.00\n"  # line 2: bad date
        + "2026-03-02,,-1.00\n"  # line 3: empty description
        + "2026-03-02,C,1.234,56\n"  # line 4: wrong field count (unquoted comma)
        + "2026-03-02,D,12.345\n"  # line 5: bad amount
        + "2026-03-02,E,-1.00\n"  # line 6: valid
        + "bad,,NaN\n"  # line 7: three errors
    )

    issues = csv_issues(text)

    assert issue_keys(issues) == [
        (2, "date"),
        (3, "description"),
        (4, None),
        (5, "amount"),
        (7, "date"),
        (7, "description"),
        (7, "amount"),
    ]
    assert issues[2].message == "expected 3 fields, found 4"


@pytest.mark.parametrize(
    "amount", ["1.234,56", "12.345", "1e3", "NaN", "abc", "", "$", "(-1.00)", "1,23"]
)
def test_rejects_invalid_amounts(amount: str) -> None:
    issues = csv_issues(HEADER + f'2026-03-02,A,"{amount}"\n')

    assert issue_keys(issues) == [(2, "amount")]


def test_error_messages_do_not_echo_cell_values() -> None:
    secret_amount, secret_date = "SECRET-AMOUNT-9876", "SECRET-DATE"

    issues = csv_issues(HEADER + f"{secret_date},A,{secret_amount}\n")

    for issue in issues:
        assert secret_amount not in issue.message
        assert secret_date not in issue.message


def test_transaction_ids_provided_must_be_unique() -> None:
    text = (
        "date,description,amount,transaction_id\n"
        "2026-03-02,A,-1.00,T-1\n"
        "2026-03-03,B,-2.00,T-1\n"
        "2026-03-04,C,-3.00,T-2\n"
    )

    issues = csv_issues(text)

    assert issue_keys(issues) == [(3, "transaction_id")]
    assert "first used on line 2" in issues[0].message


def test_generated_id_colliding_with_provided_id_is_reported() -> None:
    text = (
        "date,description,amount,transaction_id\n"
        "2026-03-02,A,-1.00,txn-000003\n"
        "2026-03-03,B,-2.00,\n"  # generated: txn-000003
    )

    assert issue_keys(csv_issues(text)) == [(3, "transaction_id")]


@pytest.mark.parametrize("bad_id", ["-leading-dash", "has space", "semi;colon", "x" * 65, "é"])
def test_rejects_invalid_provided_transaction_ids(bad_id: str) -> None:
    text = f'date,description,amount,transaction_id\n2026-03-02,A,-1.00,"{bad_id}"\n'

    assert issue_keys(csv_issues(text)) == [(2, "transaction_id")]


def test_issue_reporting_is_capped() -> None:
    text = HEADER + "bad,A,-1.00\n" * (MAX_REPORTED_ISSUES + 20)

    issues = csv_issues(text)

    assert len(issues) == MAX_REPORTED_ISSUES + 1
    assert issues[-1].row_number is None
    assert "stopped reporting" in issues[-1].message


def test_exactly_max_issues_has_no_truncation_note() -> None:
    issues = csv_issues(HEADER + "bad,A,-1.00\n" * MAX_REPORTED_ISSUES)

    assert len(issues) == MAX_REPORTED_ISSUES
    assert all(issue.row_number is not None for issue in issues)


def test_cap_overshoot_within_one_row_is_trimmed() -> None:
    rows = "bad,A,-1.00\n" * (MAX_REPORTED_ISSUES - 1) + "bad,,NaN\n"  # last row adds 3

    issues = csv_issues(HEADER + rows)

    assert len(issues) == MAX_REPORTED_ISSUES + 1
    assert "stopped reporting" in issues[-1].message


# ============================================================================= CSV: file errors


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"", "the file is empty"),
        (b"   \n\n\t\n", "the file is empty"),
        (b"\xef\xbb\xbf", "the file is empty"),
        (b"\xef\xbb\xbf\n\n", "the file is empty"),
        (HEADER.encode(), "contains no transactions"),
        ((HEADER + "\n , , \n").encode(), "contains no transactions"),
        (b"date,description,amount\n2026-03-02,\xff\xfe,-1.00\n", "must be UTF-8"),
        (b"date,description,amount\n2026-03-02,A\x00,-1.00\n", "NUL bytes"),
        (PDF, "required column is missing"),  # a PDF uploaded as the CSV
    ],
)
def test_file_level_errors(data: bytes, message: str) -> None:
    issues = csv_issues(data)

    assert any(message in issue.message for issue in issues)


def test_reports_every_missing_required_column() -> None:
    issues = csv_issues("posted,memo\n2026-03-02,A\n")

    assert issue_keys(issues) == [(1, "date"), (1, "description"), (1, "amount")]


def test_rejects_duplicate_and_blank_header_names() -> None:
    issues = csv_issues("date,description,amount,Date,\n2026-03-02,A,-1.00,x,y\n")

    assert issue_keys(issues) == [(1, "date"), (1, None)]
    assert issues[0].message == "duplicate column name"
    assert issues[1].message == "column 5 has no name"


def test_header_line_number_accounts_for_leading_blank_lines() -> None:
    issues = csv_issues("\n\nposted,description,amount\n")

    assert issue_keys(issues) == [(3, "date")]


def test_malformed_quoting_is_reported_with_line() -> None:
    issues = csv_issues(HEADER + '2026-03-02,A,-1.00\n2026-03-03,"bad"quote,-2.00\n')

    assert len(issues) == 1
    assert issues[0].row_number == 3
    assert issues[0].message.startswith("malformed CSV")


def test_unterminated_quote_is_reported() -> None:
    issues = csv_issues(HEADER + '2026-03-02,"never closed,-1.00\n')

    assert issues[0].message.startswith("malformed CSV")


def test_csv_byte_limit() -> None:
    data = csv_bytes(HEADER + "2026-03-02,A,-1.00\n")

    with pytest.raises(LimitExceededError) as excinfo:
        parse_transactions_csv(data, limits=IngestionLimits(max_csv_bytes=len(data) - 1))

    assert (excinfo.value.limit, excinfo.value.maximum) == ("csv_bytes", len(data) - 1)


def test_csv_row_limit_counts_only_data_rows() -> None:
    data = csv_bytes(HEADER + "2026-03-02,A,-1.00\n\n\n" * 4)

    with pytest.raises(LimitExceededError) as excinfo:
        parse_transactions_csv(data, limits=IngestionLimits(max_csv_rows=3))

    assert excinfo.value.limit == "csv_rows"


def test_row_limit_is_enforced_even_after_issue_cap() -> None:
    data = csv_bytes(HEADER + "bad,A,-1.00\n" * (MAX_REPORTED_ISSUES + 5))

    with pytest.raises(LimitExceededError):
        parse_transactions_csv(data, limits=IngestionLimits(max_csv_rows=MAX_REPORTED_ISSUES + 4))


def test_errors_share_base_class_and_codes() -> None:
    assert issubclass(CsvValidationError, IngestionError)
    assert issubclass(DocumentValidationError, IngestionError)
    assert issubclass(LimitExceededError, IngestionError)
    assert (CsvValidationError.code, DocumentValidationError.code, LimitExceededError.code) == (
        "invalid_csv",
        "invalid_documents",
        "limit_exceeded",
    )


# ============================================================================= limits


@pytest.mark.parametrize(
    "overrides",
    [{"max_csv_rows": 0}, {"max_documents": -1}, {"max_csv_bytes": True}, {"unknown": 1}],
)
def test_limits_validation(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        IngestionLimits.model_validate(overrides)


def test_default_limits_match_plan() -> None:
    limits = IngestionLimits()

    assert limits.max_documents == 25
    assert limits.max_document_bytes == 10 * 1024 * 1024
    assert limits.max_csv_bytes == 5 * 1024 * 1024
    assert limits.max_csv_rows == 2_000
    assert limits.max_csv_columns == 100


# ============================================================================= documents: types


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (PDF, MediaType.PDF),
        (PNG, MediaType.PNG),
        (JPEG, MediaType.JPEG),
        (b"GIF89a" + b"\x00" * 10, None),
        (b"MZ\x90\x00" + b"\x00" * 10, None),  # Windows executable
        (b"PK\x03\x04" + b"\x00" * 10, None),  # ZIP / DOCX / XLSX
        (b"<html><body>receipt</body></html>", None),
        (b"%!PS-Adobe-3.0", None),  # PostScript, not PDF
        (b" %PDF-1.7", None),  # signature must be at offset 0
        (b"\x89PNG", None),  # truncated signature
        (b"\xff\xd8", None),  # truncated signature
        (b"", None),
    ],
)
def test_detect_media_type(content: bytes, expected: MediaType | None) -> None:
    assert detect_media_type(content) is expected


def test_ingests_supported_documents() -> None:
    result = ingest_documents(
        [
            UploadedFile("receipt.pdf", PDF),
            UploadedFile("photo.png", PNG),
            UploadedFile("scan.jpeg", JPEG),
        ]
    )

    assert result.duplicates == ()
    assert [d.metadata.media_type for d in result.documents] == [
        MediaType.PDF,
        MediaType.PNG,
        MediaType.JPEG,
    ]
    first = result.documents[0]
    assert first.metadata.sha256 == hashlib.sha256(PDF).hexdigest()
    assert first.metadata.id == first.metadata.sha256[:16]
    assert first.metadata.size_bytes == len(PDF)
    assert first.metadata.filename == "receipt.pdf"
    assert first.content == PDF


def test_document_ids_are_stable_across_calls_and_names() -> None:
    first = ingest_documents([UploadedFile("a.pdf", PDF)]).documents[0]
    second = ingest_documents([UploadedFile("renamed.pdf", PDF)]).documents[0]

    assert first.metadata.id == second.metadata.id


def test_signature_decides_type_not_extension() -> None:
    """A PNG named .pdf is ingested as PNG. The claimed extension is ignored."""
    result = ingest_documents([UploadedFile("receipt.pdf", PNG)])

    assert result.documents[0].metadata.media_type is MediaType.PNG


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("receipt.pdf", b"MZ\x90\x00 this is an executable"),
        ("receipt.png", b"<script>alert(1)</script>"),
        ("receipt.jpg", b"GIF89a fake"),
        ("invoice.pdf", b"PK\x03\x04 zip archive"),
    ],
)
def test_spoofed_extensions_are_rejected(filename: str, content: bytes) -> None:
    with pytest.raises(DocumentValidationError) as excinfo:
        ingest_documents([UploadedFile(filename, content)])

    (issue,) = excinfo.value.issues
    assert issue.code is DocumentIssueCode.UNSUPPORTED_TYPE
    assert (issue.upload_index, issue.filename) == (0, filename)


def test_collects_every_rejected_upload() -> None:
    limits = IngestionLimits(max_document_bytes=len(PDF))
    uploads = [
        UploadedFile("ok.pdf", PDF),
        UploadedFile("empty.pdf", b""),
        UploadedFile("big.png", PNG + b"\x00" * len(PDF)),
        UploadedFile("notes.txt", b"hello"),
    ]

    with pytest.raises(DocumentValidationError) as excinfo:
        ingest_documents(uploads, limits=limits)

    assert [(i.upload_index, i.code) for i in excinfo.value.issues] == [
        (1, DocumentIssueCode.EMPTY_FILE),
        (2, DocumentIssueCode.FILE_TOO_LARGE),
        (3, DocumentIssueCode.UNSUPPORTED_TYPE),
    ]


def test_document_exactly_at_size_limit_is_accepted() -> None:
    result = ingest_documents(
        [UploadedFile("a.pdf", PDF)], limits=IngestionLimits(max_document_bytes=len(PDF))
    )

    assert len(result.documents) == 1


def test_no_documents_is_rejected() -> None:
    with pytest.raises(DocumentValidationError) as excinfo:
        ingest_documents([])

    assert excinfo.value.issues[0].code is DocumentIssueCode.NO_DOCUMENTS
    assert excinfo.value.issues[0].upload_index is None


def test_document_count_limit() -> None:
    uploads = [UploadedFile(f"{i}.pdf", PDF + bytes([i])) for i in range(4)]

    assert len(ingest_documents(uploads, limits=IngestionLimits(max_documents=4)).documents) == 4
    with pytest.raises(LimitExceededError) as excinfo:
        ingest_documents(uploads, limits=IngestionLimits(max_documents=3))
    assert (excinfo.value.limit, excinfo.value.maximum) == ("document_count", 3)


# ========================================================================= documents: duplicates


def test_byte_identical_uploads_are_folded_into_one_document() -> None:
    result = ingest_documents(
        [
            UploadedFile("receipt.pdf", PDF),
            UploadedFile("photo.png", PNG),
            UploadedFile("receipt (1).pdf", PDF),
            UploadedFile(None, PDF),
        ]
    )

    pdf_id = hashlib.sha256(PDF).hexdigest()[:16]
    assert [d.metadata.filename for d in result.documents] == ["receipt.pdf", "photo.png"]
    assert [(d.document_id, d.upload_index, d.filename) for d in result.duplicates] == [
        (pdf_id, 2, "receipt (1).pdf"),
        (pdf_id, 3, "document-4.pdf"),
    ]


def test_same_name_different_content_is_not_a_duplicate() -> None:
    result = ingest_documents([UploadedFile("r.pdf", PDF), UploadedFile("r.pdf", PDF + b" ")])

    assert len(result.documents) == 2
    assert result.duplicates == ()


def test_document_id_prefix_collision_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    digests = iter(["a" * 16 + "0" * 48, "a" * 16 + "1" * 48])
    monkeypatch.setattr(
        "bookkeeping_assistant.ingestion._sha256_hex", lambda content: next(digests)
    )

    with pytest.raises(IngestionError, match="collision"):
        ingest_documents([UploadedFile("a.pdf", PDF), UploadedFile("b.pdf", PDF + b"x")])


def test_content_is_not_in_repr() -> None:
    result = ingest_documents([UploadedFile("a.pdf", PDF)])

    assert "synthetic" not in repr(result)
    assert "synthetic" not in repr(UploadedFile("a.pdf", PDF))


# ============================================================================= filenames


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("receipt.pdf", "receipt.pdf"),
        ("Receipt March (1).PDF", "Receipt March (1).PDF"),
        ("../../etc/passwd", "passwd"),
        ("C:\\Users\\someone\\Desktop\\receipt.pdf", "receipt.pdf"),
        ("folder/", "fallback.pdf"),
        ("café-receipt.pdf", "cafe-receipt.pdf"),
        ("rec\x00eipt.pdf", "rec_eipt.pdf"),
        ("rec\neipt\t.pdf", "rec_eipt_.pdf"),
        ("<script>alert(1)</script>.pdf", "script_.pdf"),  # "/" splits: basename "script>.pdf"
        ("<img src=x onerror=alert(1)>.pdf", "_img src_x onerror_alert(1)_.pdf"),
        ("invoice\u202egpj.pdf", "invoicegpj.pdf"),  # right-to-left override removed
        ("a;b|c&d$e`f.pdf", "a_b_c_d_e_f.pdf"),
        ("   spaced    name.pdf  ", "spaced name.pdf"),
        (".hidden.pdf", "hidden.pdf"),
        ("..", "fallback.pdf"),
        ("...", "fallback.pdf"),
        ("", "fallback.pdf"),
        (None, "fallback.pdf"),
        ("領収書.pdf", "fallback.pdf"),  # non-Latin stem folds to nothing
        ("___", "fallback.pdf"),
        (".pdf", "fallback.pdf"),
        ("README", "README"),
    ],
)
def test_sanitize_filename(name: str | None, expected: str) -> None:
    assert sanitize_filename(name, fallback="fallback.pdf") == expected


def test_sanitize_filename_caps_length_and_keeps_extension() -> None:
    result = sanitize_filename("a" * 300 + ".pdf", fallback="x")

    assert len(result) == MAX_FILENAME_LENGTH
    assert result.endswith(".pdf")


def test_sanitize_filename_caps_length_without_extension() -> None:
    assert len(sanitize_filename("b" * 300, fallback="x")) == MAX_FILENAME_LENGTH


@pytest.mark.parametrize("name", ["../x.pdf", "a/b\\c.pdf", "\x00\x01.pdf", "x" * 500])
def test_sanitized_names_contain_no_separators_or_control_characters(name: str) -> None:
    result = sanitize_filename(name, fallback="f")

    assert "/" not in result
    assert "\\" not in result
    assert result.isprintable()
    assert not result.startswith(".")


def test_fallback_name_uses_detected_extension() -> None:
    result = ingest_documents([UploadedFile(None, PNG), UploadedFile("", JPEG)])

    assert [d.metadata.filename for d in result.documents] == ["document-1.png", "document-2.jpg"]


# ============================================================================= header issue cap


def test_blank_header_issues_are_capped() -> None:
    """Regression: every blank header cell used to produce an issue, without limit."""
    # An all-blank header is a blank line (skipped), so keep one named column:
    # 99 blank names + 2 missing required columns = 101 issues.
    header = "date" + "," * (IngestionLimits().max_csv_columns - 1)

    issues = csv_issues(header + "\n")

    assert len(issues) == MAX_REPORTED_ISSUES + 1
    assert issues[0].message == "column 2 has no name"
    assert issues[98].message == "column 100 has no name"
    assert issue_keys(issues[99:100]) == [(1, "description")]
    assert issues[-1].row_number is None
    assert "stopped reporting" in issues[-1].message


def test_duplicate_header_issues_are_capped() -> None:
    header = ",".join(["date"] * IngestionLimits().max_csv_columns)  # 99 duplicates + 2 missing

    issues = csv_issues(header + "\n")

    assert len(issues) == MAX_REPORTED_ISSUES + 1
    assert all(issue.message == "duplicate column name" for issue in issues[:99])
    assert issue_keys(issues[99:100]) == [(1, "description")]  # "amount" was cut off
    assert "stopped reporting" in issues[-1].message


def test_header_issues_below_cap_have_no_truncation_note() -> None:
    issues = csv_issues("date,date,,amount\n")

    assert issue_keys(issues) == [(1, "date"), (1, None), (1, "description")]


def test_header_with_exactly_max_columns_is_accepted() -> None:
    limits = IngestionLimits()
    extras = [f"extra_{i}" for i in range(limits.max_csv_columns - 3)]
    header = ",".join(["date", "description", "amount", *extras])
    row = ",".join(["2026-03-02", "A", "-1.00", *([""] * len(extras))])

    parsed = parse_transactions_csv(csv_bytes(f"{header}\n{row}\n"))

    assert len(parsed.transactions[0].raw) == limits.max_csv_columns


@pytest.mark.parametrize("extra_columns", [1, 5_000])
def test_header_over_max_columns_is_a_single_issue(extra_columns: int) -> None:
    limits = IngestionLimits()
    header = "date,description,amount" + ",x" * (limits.max_csv_columns - 3 + extra_columns)

    issues = csv_issues(header + "\n")

    assert len(issues) == 1
    assert issues[0].row_number == 1
    assert f"at most {limits.max_csv_columns} allowed" in issues[0].message


def test_custom_column_limit() -> None:
    limits = IngestionLimits(max_csv_columns=4)

    parse_transactions_csv(
        csv_bytes("date,description,amount,memo\n2026-03-02,A,-1,m\n"), limits=limits
    )
    issues = csv_issues("date,description,amount,memo,extra\n", limits=limits)
    assert "at most 4 allowed" in issues[0].message


def test_column_limit_cannot_be_below_required_columns() -> None:
    with pytest.raises(ValidationError):
        IngestionLimits(max_csv_columns=2)


# ================================================================ syntax error after row issues


def test_row_issues_are_kept_when_a_syntax_error_follows() -> None:
    """Regression: a malformed-quoting error used to discard every issue found before it."""
    text = (
        HEADER
        + "bad,A,-1.00\n"  # line 2: bad date
        + "2026-03-02,,-1.00\n"  # line 3: empty description
        + "2026-03-03,B,-2.00\n"  # line 4: valid
        + '2026-03-04,"bad"quote,-3.00\n'  # line 5: unrecoverable syntax error
        + "also-bad,,NaN\n"  # line 6: after the syntax error, so never read
    )

    issues = csv_issues(text)

    assert issue_keys(issues) == [(2, "date"), (3, "description"), (5, None)]
    assert issues[-1].message.startswith("malformed CSV")


def test_syntax_error_alone_is_still_reported() -> None:
    issues = csv_issues(HEADER + "2026-03-02,A,-1.00\n" + '2026-03-03,"x"y,-2.00\n')

    assert issue_keys(issues) == [(3, None)]


def test_syntax_error_fills_the_last_reporting_slot() -> None:
    text = HEADER + "bad,A,-1.00\n" * (MAX_REPORTED_ISSUES - 1) + '2026-03-03,"x"y,-2.00\n'

    issues = csv_issues(text)

    assert len(issues) == MAX_REPORTED_ISSUES  # no truncation note: nothing was dropped
    assert issues[-1].message.startswith("malformed CSV")


def test_syntax_error_after_cap_respects_the_limit() -> None:
    text = HEADER + "bad,A,-1.00\n" * MAX_REPORTED_ISSUES + '2026-03-03,"x"y,-2.00\n'

    issues = csv_issues(text)

    assert len(issues) == MAX_REPORTED_ISSUES + 1
    assert "stopped reporting" in issues[-1].message


@pytest.mark.parametrize(
    "kwargs",
    [{"date_order": DateOrder.DMY}, {"encoding": CsvEncoding.WINDOWS_1255}],
)
def test_parse_transactions_csv_rejects_unimplemented_regional_formats(
    kwargs: dict[str, Any],
) -> None:
    """Contract stub: only mdy + utf-8 are implemented until the i18n batch lands."""
    with pytest.raises(NotImplementedError):
        parse_transactions_csv(b"date,description,amount\n2026-03-02,X,-1.00\n", **kwargs)

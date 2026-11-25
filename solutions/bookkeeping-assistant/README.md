# Bookkeeping Exception Assistant

> **Status: Phase 3 of 7 (extraction schema, fake extractor, normalization).** Not yet runnable
> as a service, and there is no OpenAI integration yet.

A local tool for independent US bookkeeping firms. You give it one client's bank-transaction
export plus supporting documents (receipts, invoices). It returns a list of transactions and
documents that **need human review**:

- possibly missing receipts
- amount mismatches between a transaction and its document
- possible duplicate documents
- ambiguous transaction-to-document matches
- missing or incomplete document information

## What it is not

- **Not an accounting, audit, tax, or legal authority.** Every finding is a suggestion for a
  human to review.
- **It never modifies accounting records.** Nothing is written back to any accounting system.
- Thresholds such as the $75 missing-receipt default are configurable heuristics, not tax
  guidance.

## Design principles

- **LLM extracts, code decides.** An LLM reads documents into raw text fields (Phase 6). All
  money parsing, matching, and rule checks are deterministic Python.
- **Money is `Decimal`, never `float`.** Amounts serialize to JSON as strings (`"12.30"`).
  Ambiguous formats such as `1.234,56` are rejected, never guessed.
- **Every finding carries evidence**, labelled by provenance: bank export, AI extraction, or
  computed.

## Development

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12+.

```bash
cd solutions/bookkeeping-assistant
uv sync                       # create .venv and install dependencies

uv run pytest                 # tests
uv run ruff check .           # lint
uv run ruff format --check .  # formatting
uv run mypy src tests         # type check (strict mode, configured in pyproject.toml)
```

## Layout (current)

```
src/bookkeeping_assistant/
  money.py    strict amount parsing (parse_amount) and the Money type
  models.py   domain models and enums (transactions, documents, matches, findings, evidence)
  config.py   ReviewConfig: thresholds, date windows, scoring weights
  ingestion.py  transactions CSV parsing, upload validation (signatures, limits), hashing
  extraction/
    schema.py   strict extraction contract (raw strings + quotes) and ExtractionResult
    base.py     Extractor protocol and FakeExtractor (offline, fixtures keyed by SHA-256)
  normalization.py  deterministic parsing of extracted fields into NormalizedDocument
tests/unit/   unit tests for the above
tests/fixtures/  synthetic test data only
```

## Transactions CSV format

UTF-8 (a BOM is fine), with a header row. Column names are case-insensitive; order does not matter.

| Column | Required | Format |
|---|---|---|
| `date` | yes | `YYYY-MM-DD` or `MM/DD/YYYY` (US month-first) |
| `description` | yes | non-empty text |
| `amount` | yes | US format; negative = money out. `$`, `1,234.56`, and `(12.00)` accepted |
| `transaction_id` | no | 1-64 chars: letters, digits, `. _ : -`. Must be unique. If blank, an ID is derived from the line number |

Extra columns are kept as evidence but otherwise ignored. Limits: 5 MB, 2,000 rows. Documents: PDF, PNG, or JPEG (detected by file signature), up to 25 files of 10 MB each. Byte-identical uploads are processed once and reported as duplicates.

## Data handling

Use **synthetic data only** in this repository. Never commit real client exports, receipts, or
reports. `local-data/` is git-ignored for local experiments.

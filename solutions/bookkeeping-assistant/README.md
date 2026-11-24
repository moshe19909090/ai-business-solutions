# Bookkeeping Exception Assistant

> **Status: Phase 1 of 7 (scaffold, domain models, money parsing).** Not yet runnable as a service.

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
tests/unit/   unit tests for the above
```

## Data handling

Use **synthetic data only** in this repository. Never commit real client exports, receipts, or
reports. `local-data/` is git-ignored for local experiments.

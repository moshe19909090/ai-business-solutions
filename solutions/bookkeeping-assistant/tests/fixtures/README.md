# Test fixtures

**Synthetic data only.** Never add real client exports, receipts, or invoices here.

- `transactions_valid.csv`: a canonical transactions CSV.
- `documents/`: tiny placeholder files that carry valid PDF, PNG, or JPEG signatures. They are
  not real documents. They exist so their SHA-256 can key a fake extraction.
- `extractions/<sha256>.json`: what `FakeExtractor` returns for the document with that hash.
  The directory must contain only these files.

| Document | Extraction fixture | Scenario |
|---|---|---|
| `receipt-coffee.pdf` | `6a0fd56d….json` | Complete USD receipt (date includes a time) |
| `invoice-office-supply.pdf` | `dc866cf5….json` | Invoice, month-name date, no currency printed |
| `receipt-unreadable.png` | `4d1890ef….json` | Extractor reports the document as unreadable |
| `receipt-eur.jpg` | `99fc0363….json` | Non-USD receipt with a European-format total |
| `receipt-invalid-output.pdf` | `8d0d523c….json` | Output violates the schema (number value, unknown key) |
| `receipt-no-fixture.pdf` | none | No fixture, so `FakeExtractor` returns `no_fixture` |

If you change a document, its hash changes. Rename its extraction fixture to match.
`test_fixture_extractions_match_fixture_documents` catches mismatches.

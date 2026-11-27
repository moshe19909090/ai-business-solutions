# Shared contracts: Phase 5 and the multilingual batch

This document is the T0 contract that the four implementation agents code against. The lead
owns it, together with `models.py`, `config.py`, `profiles.py` and `messages.py`. If you need a
change to any of these, ask the lead: they make it as a follow-up contract commit, and nobody
else edits these files.

## 1. Concerns kept separate

| Concern | Where | Rule |
|---|---|---|
| Text and script | `text.py` (Agent B) | Language-agnostic. Script is detected from Unicode data and used only to compare names. |
| Regional format | `profiles.RegionalFormat` | Covers `date_order` (`mdy`/`dmy`), `csv_encoding` (`utf-8`/`utf-16`/`windows-1255`) and `number_format` (`1,234.56` only). The user declares it; it is never guessed. |
| Account | `profiles.AccountProfile` | `currency` is in `SUPPORTED_ACCOUNT_CURRENCIES` = {USD, ILS}. Every transaction is in this currency. |
| Report language | `messages.Locale` (`en`, `he`) | Changes rendered text only. It never affects parsing, matching or findings. |

Hebrew text never implies ILS or day-first dates. A document's language never selects a
regional format.

`ReviewConfig.account` and `ReviewConfig.regional` hold the profiles, and both are echoed in
every report. The money thresholds `missing_document_threshold` and `high_severity_delta` are in
the account currency. For any account other than USD, both must be set explicitly; there are no
ILS defaults.

**No cross-currency comparison.** The rules depend on the document's currency:
- Same as the account currency: the document is eligible for matching.
- Unknown: the document is eligible, with the `currency_unconfirmed` assumption and hint.
- Different: the document is excluded from matching, and an INCOMPLETE_DOCUMENT finding is raised.

Nothing is ever converted.

## 2. Signatures

### Agent B (i18n core)

Unchanged:
- `normalize_vendor(text: str) -> str`. Returns `""` only when no letter of any script remains.
- `match_transactions(transactions, documents, config) -> MatchingOutcome`

New keyword arguments, each with a default that keeps today's behaviour:
- `normalize_extraction(result, *, date_order: DateOrder = DateOrder.MDY) -> NormalizedDocument`

T0 adds both keyword signatures as stubs that raise `NotImplementedError` for non-default values, so every branch compiles against them. T0 also renames `DocumentExclusion.CURRENCY_NOT_USD` → `CURRENCY_MISMATCH`. B replaces the stubs with the real behaviour and keeps the signatures.
- `parse_transactions_csv(data, *, limits=DEFAULT_LIMITS, date_order: DateOrder = DateOrder.MDY, encoding: CsvEncoding = CsvEncoding.UTF8)`

Matching obligations:
- Compare a document's currency with `config.account.currency`, not with a constant.
- Keep the exclusion values final: `unreadable`, `total_unavailable`, `total_not_positive`, `date_unavailable` and `currency_mismatch` (renamed in T0). `messages.DOCUMENT_EXCLUSION_CODES` mirrors them, and a test keeps the two in sync.
- Fill `MatchResult.tied_document_ids` for **every** ambiguous result. The model accepts an empty tuple only until B lands; integration then makes it required.
- Record `MatchAssumption.VENDOR_NOT_COMPARABLE` when two vendor names share no script. Never auto-assign a candidate that relies on it.
- Stop emitting `NormalizationIssue.CURRENCY_UNSUPPORTED` (deprecated).

`text.py` public API:

| Function | Purpose |
|---|---|
| `strip_bidi_controls(text) -> str` | Remove bidi control characters |
| `normalize_for_display(text) -> str \| None` | NFC, controls removed, ZWJ/ZWNJ kept |
| `normalize_for_matching(text) -> str` | Build the comparison form of a name |
| `has_letter(text) -> bool` | True if any letter of any script is present |
| `scripts(text) -> frozenset[Script]` | Scripts used in the text |
| `scripts_overlap(a, b) -> bool` | True if the two texts share a script |

### Agent A (rules and report)

```python
# validation.py
@dataclass(frozen=True, slots=True)
class RuleContext: ...            # transactions, sources, extractions, documents, matching,
                                  # duplicate_uploads, config; RuleContext.build(...) checks ID sets
def run_rules(ctx: RuleContext) -> tuple[Finding, ...]      # sorted; raises on an ID collision

# review.py
def build_review_report(
    transactions: ParsedTransactions,
    documents: Sequence[SourceDocument],
    duplicate_uploads: Sequence[DuplicateUpload],
    extractions: Sequence[ExtractionResult],
    config: ReviewConfig,
    *,
    report_id: str,
    generated_at: dt.datetime,        # timezone-aware UTC
    locale: Locale = Locale.EN,
) -> ReviewReport
```

`ReviewReport` lives in `review.py`. Its fields are:
- `schema_version` (`"1"`), `report_id`, `generated_at`
- `locale`, plus `direction` (`ltr`/`rtl`, from `messages.text_direction`)
- `disclaimer`, rendered from `report.disclaimer`
- `config`, which includes `account` and `regional`
- `inputs` (CSV SHA-256, transaction count, source documents, duplicate uploads)
- `extractions`, `documents`, `matches`, `excluded_documents`, `findings`
- `summary`: counts per finding type and per severity, with zeros included

Two reports built from the same inputs are identical apart from `report_id` and
`generated_at`. The language of the report changes only the rendered text: `title`,
`explanation`, evidence `label`, coded evidence `value`, and `disclaimer`. Findings, IDs and
order stay the same.

## 3. Findings

| Field | Contract |
|---|---|
| `type` | `FindingType` |
| `message` | `FindingMessage`, which must belong to `type` |
| `params` | Exactly the names in `FINDING_MESSAGE_PARAMS[message]`, values are locale-neutral tokens (see below) |
| `title`, `explanation` | `render(finding.<message>.title / .explanation, params, locale=...)` |
| `hints` | Unique `FindingHint`s (`tip_possible`, `lower_bank_amount`, `currency_unconfirmed`, `date_ambiguous`, `vendor_not_comparable`) |
| `next_action` | One `NextAction` |
| `rule` | `RuleRef`. `rule_id` is `missing_document`, `amount_mismatch`, `duplicate_document.exact`, `duplicate_document.probable`, `ambiguous_match`, `incomplete_document` or `unmatched_document`. `rule_version` is `"1"`. `parameters` are the config values the rule read, as `str()` |

**Param values must be locale-neutral tokens.** Each is 1–64 characters drawn from ASCII letters,
digits and `. _ : + -`. This covers amounts (`-12.40`), ISO currency codes, ISO dates, counts,
ratios and IDs. It deliberately rejects free text: a vendor name, description, filename or quote
cannot be a parameter, so it can only appear in evidence.

| Message | Params |
|---|---|
| `missing_document` | `amount`, `currency`, `date`, `threshold` |
| `amount_mismatch` | `bank_amount`, `document_total`, `difference`, `currency` |
| `duplicate_document_exact` | `copies` |
| `duplicate_document_probable` | `count` |
| `ambiguous_match` | `count` |
| `incomplete_document` | `count` |
| `unmatched_document` | (none) |

**Severity by rule:**

| Rule | Severity |
|---|---|
| Missing document | medium |
| Amount mismatch | medium; high above `high_severity_delta` or `high_severity_ratio` |
| Exact duplicate | high |
| Probable duplicate | medium |
| Ambiguous match | medium |
| Incomplete document | high if extraction failed or the document is unreadable; medium if the document was excluded from matching; low otherwise |
| Unmatched document | info |

An ambiguous date never creates its own finding. It shows up as the `date_ambiguous` hint on
the findings that use that date.

**Finding ID:**

```
"F-" + sha256(json.dumps([rule_id, sorted(transaction_ids), sorted(document_ids)],
    ensure_ascii=True, separators=(",", ":"))).hexdigest()[:16]
```

**Order of findings:**

1. Severity: high, medium, low, info
2. `FindingType` declaration order
3. Lowest CSV row of the finding's transactions (findings without transactions last)
4. `document_ids`
5. `id`

Inside a finding:
- `transaction_ids` are in CSV row order.
- `document_ids` are sorted.
- Evidence goes bank, then upload, then AI, then computed.

## 4. Evidence

| Field | Contract |
|---|---|
| `provenance` | `bank_export`, `upload`, `ai_extraction` or `computed` |
| `field` | An `EvidenceField` code. The label comes from the catalog key `evidence.<field>` |
| `value` | Display form, in the original script: NFC, bidi controls removed. Money is a plain decimal string with `currency` set. Never put `$` or any other symbol in `value`. |
| `raw_value` | The verbatim CSV cell or extracted value, set only when it differs from `value` |
| `code` | Set when the value is a code (a `NormalizationIssue`, an exclusion reason, an extraction status). In that case `value` is the rendered catalog text for the code, from `issue.<code>` and so on |
| `source_ref` | `csv:row:<n>`, `csv:row:<n>:col:<name>`, `upload:<index>`, `doc:<id>:file`, `doc:<id>:<field>`, `computed:<name>` or `match:<txn>:<doc>` |
| `quote`, `page`, `model` | Allowed only for `ai_extraction` evidence |

## 5. Message catalogs (Agent D)

`locales/<locale>.json` has the form `{"locale", "direction", "messages": {key: template}}`.

**Keys** (see `messages.required_keys()`):
- `finding.<message>.title`, `finding.<message>.explanation`
- `hint.*`, `next_action.*`, `severity.*`, `evidence.*`, `issue.*`
- `report.disclaimer`
- `exclusion.<DocumentExclusion>`

**Templates:**
- `{name}` placeholders only.
- They may use only the contract params.
- Every locale must define every required key, with the same placeholders as English.
- In an RTL catalog, each substituted value is automatically wrapped in a left-to-right isolate.

**Language rules:**
- Never use "error", "wrong", "fraud", "suspicious" or "confidence %".
- Say "possible", "may", "needs review" and "AI-read".
- Severity is a review priority, not a risk level.

## 6. File ownership (batch 1)

| Owner | Files |
|---|---|
| Lead | `models.py`, `config.py`, `profiles.py`, `messages.py`, `pyproject.toml`, `docs/contracts.md`, `tests/unit/test_{models,config,profiles,message_rendering}.py` |
| A | `validation.py`, `review.py`, `tests/unit/test_{validation,review}.py` |
| B | `text.py`, `normalization.py`, `matching.py`, `ingestion.py`, `money.py`, `tests/unit/test_{text,normalization,matching,ingestion,money}.py` |
| C | `tests/acceptance/**`, `tests/integration/**`, `tests/fixtures/{multilingual,golden}/**`, root `.gitattributes`, one pointer line in `tests/fixtures/README.md` |
| D | `locales/en.json`, `locales/he.json`, `tests/unit/test_messages.py`, `docs/report-contract.md`, `docs/rtl-ui-requirements.md`, solution `README.md` |

**Testing conventions:**
- No root `tests/conftest.py`.
- Test file names must be unique across directories.
- No new pytest markers.
- Ruff's confusable-character checks (RUF001/RUF003) are off in `tests/**` only. In `src`, non-ASCII literals must be deliberate, or written as `\u` escapes.

import datetime as dt
import hashlib
from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from bookkeeping_assistant.config import ScoreWeights
from bookkeeping_assistant.models import (
    FINDING_MESSAGE_PARAMS,
    AmbiguityReason,
    BankTransaction,
    Direction,
    DocumentType,
    Evidence,
    EvidenceField,
    Finding,
    FindingHint,
    FindingMessage,
    FindingType,
    MatchAssumption,
    MatchCandidate,
    MatchResult,
    MatchStatus,
    MediaType,
    NextAction,
    NormalizationIssue,
    NormalizedDocument,
    Provenance,
    RuleRef,
    ScoreBreakdown,
    Severity,
    SourceDocument,
)

SHA = hashlib.sha256(b"synthetic receipt").hexdigest()


# ----------------------------------------------------------------------------- builders


def make_transaction(**overrides: Any) -> BankTransaction:
    data: dict[str, Any] = {
        "id": "txn-0002",
        "row_number": 2,
        "posted_date": dt.date(2026, 3, 4),
        "description": "SQ *SAMPLE COFFEE",
        "amount": Decimal("-12.40"),
        "raw": {"date": "2026-03-04", "description": "SQ *SAMPLE COFFEE", "amount": "-12.40"},
    }
    data.update(overrides)
    return BankTransaction.model_validate(data)


def make_breakdown(**overrides: Any) -> ScoreBreakdown:
    data: dict[str, Any] = {
        "amount_delta": Decimal("0.00"),
        "amount_component": 0.5,
        "date_delta_days": 1,
        "date_component": 0.17,
        "vendor_similarity": 0.9,
        "vendor_component": 0.27,
        "total": 0.94,
    }
    data.update(overrides)
    return ScoreBreakdown.model_validate(data)


def make_candidate(document_id: str, transaction_id: str = "txn-0002") -> MatchCandidate:
    return MatchCandidate(
        transaction_id=transaction_id, document_id=document_id, breakdown=make_breakdown()
    )


def make_evidence(**overrides: Any) -> Evidence:
    data: dict[str, Any] = {
        "provenance": Provenance.BANK_EXPORT,
        "field": EvidenceField.TRANSACTION_AMOUNT,
        "label": "Bank amount",
        "value": "-12.40",
        "currency": "USD",
        "source_ref": "csv:row:2",
    }
    data.update(overrides)
    return Evidence.model_validate(data)


def make_finding(**overrides: Any) -> Finding:
    data: dict[str, Any] = {
        "id": "fnd-0001",
        "type": FindingType.MISSING_DOCUMENT,
        "severity": Severity.MEDIUM,
        "message": FindingMessage.MISSING_DOCUMENT,
        "params": {
            "amount": "-120.00",
            "currency": "USD",
            "date": "2026-03-04",
            "threshold": "75.00",
        },
        "title": "Possibly missing document",
        "explanation": "No uploaded document was matched to this outflow.",
        "next_action": NextAction.REQUEST_DOCUMENT,
        "transaction_ids": ("txn-0002",),
        "evidence": (make_evidence(),),
        "rule": RuleRef(
            rule_id="MISSING_DOCUMENT", rule_version="1", parameters={"threshold": "75.00"}
        ),
    }
    data.update(overrides)
    return Finding.model_validate(data)


# ----------------------------------------------------------------------------- shared behaviour


def test_models_are_frozen() -> None:
    transaction = make_transaction()

    with pytest.raises(ValidationError):
        transaction.amount = Decimal("1.00")  # type: ignore[misc]


def test_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        make_transaction(vendor_normalized="SAMPLE COFFEE")


# ----------------------------------------------------------------------------- deep immutability


def _rule_ref(parameters: dict[str, str] | None = None) -> RuleRef:
    if parameters is None:
        return RuleRef(rule_id="MISSING_DOCUMENT", rule_version="1")
    return RuleRef(rule_id="MISSING_DOCUMENT", rule_version="1", parameters=parameters)


@pytest.mark.parametrize(
    "get_map",
    [
        pytest.param(lambda: make_transaction().raw, id="BankTransaction.raw"),
        pytest.param(lambda: _rule_ref({"threshold": "75.00"}).parameters, id="RuleRef.parameters"),
        pytest.param(lambda: _rule_ref().parameters, id="RuleRef.parameters-default"),
        pytest.param(lambda: make_finding().rule.parameters, id="Finding.rule.parameters"),
    ],
)
def test_string_maps_are_read_only(get_map: Any) -> None:
    mapping = get_map()

    with pytest.raises(TypeError):
        mapping["amount"] = "999.99"
    with pytest.raises(TypeError):
        del mapping[next(iter(mapping), "missing")]
    assert not hasattr(mapping, "update")
    assert not hasattr(mapping, "clear")


def test_string_maps_are_copied_from_caller_input() -> None:
    raw = {"date": "2026-03-04", "description": "SQ *SAMPLE COFFEE", "amount": "-12.40"}
    parameters = {"threshold": "75.00"}
    transaction = make_transaction(raw=raw)
    rule = _rule_ref(parameters)

    raw["amount"] = "-999.99"
    parameters["threshold"] = "0.00"

    assert transaction.raw["amount"] == "-12.40"
    assert rule.parameters["threshold"] == "75.00"


def test_string_maps_serialize_as_plain_objects_and_round_trip() -> None:
    transaction = make_transaction()
    finding = make_finding()

    transaction_payload = transaction.model_dump(mode="json")
    python_payload = finding.model_dump()

    assert type(transaction_payload["raw"]) is dict
    assert transaction_payload["raw"]["amount"] == "-12.40"
    assert type(python_payload["rule"]["parameters"]) is dict
    assert '"parameters":{"threshold":"75.00"}' in finding.model_dump_json()
    assert BankTransaction.model_validate_json(transaction.model_dump_json()) == transaction
    assert Finding.model_validate_json(finding.model_dump_json()) == finding
    assert Finding.model_validate(python_payload) == finding


def test_dumped_maps_are_independent_copies() -> None:
    transaction = make_transaction()

    dumped = transaction.model_dump()
    dumped["raw"]["amount"] = "-999.99"

    assert transaction.raw["amount"] == "-12.40"


@pytest.mark.parametrize("bad", [{"amount": 12}, {"amount": None}, ["amount", "-12.40"]])
def test_string_maps_still_validate_values(bad: Any) -> None:
    with pytest.raises(ValidationError):
        make_transaction(raw=bad)


# ----------------------------------------------------------------------------- BankTransaction


@pytest.mark.parametrize(
    ("amount", "direction"),
    [
        (Decimal("-12.40"), Direction.OUTFLOW),
        (Decimal("12.40"), Direction.INFLOW),
        (Decimal("0.00"), Direction.NEUTRAL),
        (Decimal("-0.00"), Direction.NEUTRAL),
        (Decimal("0.01"), Direction.INFLOW),
        (Decimal("-0.01"), Direction.OUTFLOW),
    ],
)
def test_transaction_direction_derives_from_sign(amount: Decimal, direction: Direction) -> None:
    assert make_transaction(amount=amount).direction is direction


@pytest.mark.parametrize(
    "overrides",
    [
        {"row_number": 0},
        {"id": ""},
        {"description": ""},
        {"amount": -12.4},  # float
        {"amount": Decimal("-12.405")},
        {"posted_date": "03/04/2026"},  # model takes ISO dates only; CSV parsing is Phase 2
    ],
)
def test_transaction_rejects_invalid(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        make_transaction(**overrides)


def test_transaction_json_round_trip() -> None:
    transaction = make_transaction()

    payload = transaction.model_dump(mode="json")
    restored = BankTransaction.model_validate_json(transaction.model_dump_json())

    assert payload["amount"] == "-12.40"
    assert payload["posted_date"] == "2026-03-04"
    assert "direction" not in payload  # derived, not stored
    assert restored == transaction


# ----------------------------------------------------------------------------- SourceDocument


def test_source_document_valid() -> None:
    document = SourceDocument(
        id=SHA[:16], filename="receipt.pdf", media_type=MediaType.PDF, size_bytes=10, sha256=SHA
    )

    assert document.model_dump(mode="json")["media_type"] == "application/pdf"


@pytest.mark.parametrize(
    "overrides",
    [
        {"id": "0" * 16},  # does not match hash
        {"sha256": SHA.upper()},
        {"sha256": SHA[:-1]},
        {"size_bytes": 0},
        {"media_type": "image/gif"},
        {"filename": ""},
    ],
)
def test_source_document_rejects_invalid(overrides: dict[str, Any]) -> None:
    data: dict[str, Any] = {
        "id": SHA[:16],
        "filename": "receipt.pdf",
        "media_type": "application/pdf",
        "size_bytes": 10,
        "sha256": SHA,
    }
    data.update(overrides)

    with pytest.raises(ValidationError):
        SourceDocument.model_validate(data)


# ----------------------------------------------------------------------------- NormalizedDocument


def test_normalized_document_allows_missing_fields() -> None:
    document = NormalizedDocument(
        document_id="doc-1",
        document_type=DocumentType.UNREADABLE,
        issues=(NormalizationIssue.TOTAL_MISSING, NormalizationIssue.DATE_MISSING),
    )

    assert document.total is None
    assert document.model_dump(mode="json")["issues"] == ["total_missing", "date_missing"]


def test_normalized_document_rejects_unknown_issue_codes() -> None:
    with pytest.raises(ValidationError):
        NormalizedDocument.model_validate(
            {"document_id": "doc-1", "document_type": "receipt", "issues": ["made_up_issue"]}
        )


def test_normalized_document_money_fields_are_strict() -> None:
    with pytest.raises(ValidationError):
        NormalizedDocument.model_validate(
            {"document_id": "doc-1", "document_type": "receipt", "total": 12.5}
        )


@pytest.mark.parametrize("currency", ["usd", "US", "USDX", "$"])
def test_normalized_document_rejects_bad_currency_code(currency: str) -> None:
    with pytest.raises(ValidationError):
        NormalizedDocument(
            document_id="doc-1", document_type=DocumentType.RECEIPT, currency=currency
        )


# ----------------------------------------------------------------------------- matching


@pytest.mark.parametrize(
    "field",
    ["amount_component", "date_component", "vendor_similarity", "vendor_component", "total"],
)
@pytest.mark.parametrize("value", [-0.01, 1.01, 1.0 + 1e-6, float("nan"), float("inf"), True])
def test_score_breakdown_scores_bounded(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        make_breakdown(**{field: value})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.33 + 0.56 + 0.11, 1.0),  # == 1.0000000000000002 in floating point
        (1.0 + 1e-12, 1.0),
        (-1e-12, 0.0),
        (0.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.5),
    ],
)
def test_score_clamps_insignificant_rounding_error(value: float, expected: float) -> None:
    assert make_breakdown(total=value).total == expected


def test_score_breakdown_accepts_total_from_valid_weights() -> None:
    """Regression: these weights pass ScoreWeights, but their float sum exceeds 1.0."""
    weights = ScoreWeights(amount_exact=0.33, amount_close=0.2, date=0.56, vendor=0.11)
    total = weights.amount_exact + weights.date + weights.vendor
    assert total > 1.0  # precondition: the float sum really is 1.0000000000000002

    breakdown = make_breakdown(
        amount_component=weights.amount_exact,
        date_component=weights.date,
        vendor_similarity=1.0,
        vendor_component=weights.vendor,
        total=total,
    )

    assert breakdown.total == 1.0


def test_match_result_matched_valid() -> None:
    result = MatchResult(
        transaction_id="txn-0002",
        status=MatchStatus.MATCHED,
        document_id="doc-a",
        candidates=(make_candidate("doc-a"),),
    )

    assert result.document_id == "doc-a"


def test_match_result_ambiguous_multiple_documents_valid() -> None:
    result = MatchResult(
        transaction_id="txn-0002",
        status=MatchStatus.AMBIGUOUS,
        candidates=(make_candidate("doc-a"), make_candidate("doc-b")),
        ambiguity_reasons=(AmbiguityReason.MULTIPLE_DOCUMENTS,),
    )

    assert result.document_id is None


def test_match_result_ambiguous_shared_document_with_single_candidate() -> None:
    """Regression: two transactions near-tied for one document. Each has a single candidate."""
    result = MatchResult(
        transaction_id="txn-0002",
        status=MatchStatus.AMBIGUOUS,
        candidates=(make_candidate("doc-a"),),
        ambiguity_reasons=(AmbiguityReason.SHARED_DOCUMENT,),
        competing_candidates=(make_candidate("doc-a", transaction_id="txn-0003"),),
    )

    assert [c.transaction_id for c in result.competing_candidates] == ["txn-0003"]
    assert MatchResult.model_validate_json(result.model_dump_json()) == result


def test_match_result_ambiguous_both_reasons() -> None:
    result = MatchResult(
        transaction_id="txn-0002",
        status=MatchStatus.AMBIGUOUS,
        candidates=(make_candidate("doc-a"), make_candidate("doc-b")),
        ambiguity_reasons=(AmbiguityReason.MULTIPLE_DOCUMENTS, AmbiguityReason.SHARED_DOCUMENT),
        competing_candidates=(
            make_candidate("doc-a", transaction_id="txn-0003"),
            make_candidate("doc-a", transaction_id="txn-0004"),
        ),
    )

    payload = result.model_dump(mode="json")

    assert payload["ambiguity_reasons"] == ["multiple_documents", "shared_document"]
    assert len(payload["competing_candidates"]) == 2


def test_match_result_unmatched_valid() -> None:
    assert MatchResult(transaction_id="txn-0002", status=MatchStatus.UNMATCHED).candidates == ()


def test_match_result_unmatched_keeps_candidates_lost_to_other_transactions() -> None:
    result = MatchResult(
        transaction_id="txn-0002",
        status=MatchStatus.UNMATCHED,
        candidates=(make_candidate("doc-a"),),
    )

    assert result.document_id is None
    assert MatchResult.model_validate_json(result.model_dump_json()) == result


def test_match_candidate_assumptions() -> None:
    candidate = MatchCandidate(
        transaction_id="txn-0002",
        document_id="doc-a",
        breakdown=make_breakdown(),
        assumptions=(MatchAssumption.CURRENCY_UNCONFIRMED, MatchAssumption.DATE_AMBIGUOUS),
    )

    assert candidate.model_dump(mode="json")["assumptions"] == [
        "currency_unconfirmed",
        "date_ambiguous",
    ]
    assert make_candidate("doc-a").assumptions == ()
    with pytest.raises(ValidationError, match="assumptions must not contain duplicates"):
        MatchCandidate(
            transaction_id="txn-0002",
            document_id="doc-a",
            breakdown=make_breakdown(),
            assumptions=(MatchAssumption.DATE_AMBIGUOUS, MatchAssumption.DATE_AMBIGUOUS),
        )


_MULTIPLE = (AmbiguityReason.MULTIPLE_DOCUMENTS,)
_SHARED = (AmbiguityReason.SHARED_DOCUMENT,)


@pytest.mark.parametrize(
    ("status", "document_id", "candidate_ids", "reasons", "competitors", "message"),
    [
        (MatchStatus.MATCHED, None, ("doc-a",), (), (), "must reference one of its candidates"),
        (MatchStatus.MATCHED, "doc-z", ("doc-a",), (), (), "must reference one of its candidates"),
        (MatchStatus.MATCHED, "doc-a", ("doc-a",), _SHARED, (), "only allowed when ambiguous"),
        (
            MatchStatus.MATCHED,
            "doc-a",
            ("doc-a",),
            (),
            (("txn-0003", "doc-a"),),
            "only allowed when ambiguous",
        ),
        (
            MatchStatus.AMBIGUOUS,
            "doc-a",
            ("doc-a", "doc-b"),
            _MULTIPLE,
            (),
            "document_id must be empty",
        ),
        (MatchStatus.AMBIGUOUS, None, ("doc-a", "doc-b"), (), (), "at least one ambiguity reason"),
        (MatchStatus.AMBIGUOUS, None, ("doc-a",), _MULTIPLE, (), "at least two candidates"),
        (MatchStatus.AMBIGUOUS, None, (), _SHARED, (), "at least one candidate"),
        (MatchStatus.AMBIGUOUS, None, ("doc-a",), _SHARED, (), "requires competing_candidates"),
        (
            MatchStatus.AMBIGUOUS,
            None,
            ("doc-a", "doc-b"),
            _MULTIPLE,
            (("txn-0003", "doc-a"),),
            "requires competing_candidates, and vice versa",
        ),
        (
            MatchStatus.AMBIGUOUS,
            None,
            ("doc-a", "doc-b"),
            _MULTIPLE + _MULTIPLE,
            (),
            "ambiguity_reasons must not contain duplicates",
        ),
        (
            MatchStatus.AMBIGUOUS,
            None,
            ("doc-a",),
            _SHARED,
            (("txn-0002", "doc-a"),),
            "must belong to other transactions",
        ),
        (
            MatchStatus.AMBIGUOUS,
            None,
            ("doc-a",),
            _SHARED,
            (("txn-0003", "doc-z"),),
            "must reference this result's candidates",
        ),
        (
            MatchStatus.AMBIGUOUS,
            None,
            ("doc-a",),
            _SHARED,
            (("txn-0003", "doc-a"), ("txn-0003", "doc-a")),
            "competing_candidates must not contain duplicates",
        ),
        (
            MatchStatus.AMBIGUOUS,
            None,
            ("doc-a", "doc-a"),
            _MULTIPLE,
            (),
            "distinct documents",
        ),
        (MatchStatus.UNMATCHED, "doc-a", ("doc-a",), (), (), "must have no document_id"),
        (MatchStatus.UNMATCHED, None, (), _SHARED, (), "only allowed when ambiguous"),
    ],
)
def test_match_result_rejects_inconsistent_status(
    status: MatchStatus,
    document_id: str | None,
    candidate_ids: tuple[str, ...],
    reasons: tuple[AmbiguityReason, ...],
    competitors: tuple[tuple[str, str], ...],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        MatchResult(
            transaction_id="txn-0002",
            status=status,
            document_id=document_id,
            candidates=tuple(make_candidate(doc_id) for doc_id in candidate_ids),
            ambiguity_reasons=reasons,
            competing_candidates=tuple(
                make_candidate(doc_id, transaction_id=txn_id) for txn_id, doc_id in competitors
            ),
        )


def test_match_result_rejects_candidate_of_other_transaction() -> None:
    with pytest.raises(ValidationError, match="must belong to this transaction"):
        MatchResult(
            transaction_id="txn-0002",
            status=MatchStatus.MATCHED,
            document_id="doc-a",
            candidates=(make_candidate("doc-a", transaction_id="txn-0003"),),
        )


# ----------------------------------------------------------------------------- findings


def test_finding_defaults_to_needs_review_and_serializes() -> None:
    finding = make_finding()

    payload = finding.model_dump(mode="json")

    assert finding.status == "needs_review"
    assert payload["type"] == "missing_document"
    assert payload["severity"] == "medium"
    assert payload["evidence"][0]["provenance"] == "bank_export"
    assert Finding.model_validate_json(finding.model_dump_json()) == finding


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"evidence": ()}, "at least 1 item"),
        ({"transaction_ids": (), "document_ids": ()}, "at least one transaction or document"),
        ({"status": "resolved"}, "needs_review"),
        ({"title": ""}, "at least 1 character"),
    ],
)
def test_finding_rejects_invalid(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_finding(**overrides)


def test_enum_values_are_stable() -> None:
    """Enum values are part of the API contract. Changing them is a breaking change."""
    assert [t.value for t in FindingType] == [
        "missing_document",
        "amount_mismatch",
        "duplicate_document",
        "ambiguous_match",
        "incomplete_document",
        "unmatched_document",
    ]
    assert [s.value for s in Severity] == ["high", "medium", "low", "info"]
    assert [p.value for p in Provenance] == ["bank_export", "ai_extraction", "computed", "upload"]
    assert [m.value for m in FindingMessage] == [
        "missing_document",
        "amount_mismatch",
        "duplicate_document_exact",
        "duplicate_document_probable",
        "ambiguous_match",
        "incomplete_document",
        "unmatched_document",
    ]
    assert [h.value for h in FindingHint] == [
        "tip_possible",
        "lower_bank_amount",
        "currency_unconfirmed",
        "date_ambiguous",
        "vendor_not_comparable",
    ]
    assert [a.value for a in NextAction] == [
        "request_document",
        "confirm_amount_difference",
        "confirm_duplicate",
        "choose_matching_document",
        "check_document_fields",
        "upload_clearer_copy",
        "check_other_account_or_period",
    ]
    assert [a.value for a in MatchAssumption] == [
        "currency_unconfirmed",
        "date_ambiguous",
        "vendor_not_comparable",
    ]


# ----------------------------------------------------------------------------- finding contract


def test_every_message_maps_to_its_finding_type_and_has_a_param_contract() -> None:
    assert set(FINDING_MESSAGE_PARAMS) == set(FindingMessage)
    assert {m.finding_type for m in FindingMessage} == set(FindingType)
    assert FindingMessage.DUPLICATE_DOCUMENT_EXACT.finding_type is FindingType.DUPLICATE_DOCUMENT
    assert FindingMessage.DUPLICATE_DOCUMENT_PROBABLE.finding_type is FindingType.DUPLICATE_DOCUMENT


def test_finding_serializes_codes_and_params() -> None:
    finding = make_finding(hints=(FindingHint.DATE_AMBIGUOUS,))

    payload = finding.model_dump(mode="json")

    assert payload["message"] == "missing_document"
    assert payload["params"]["threshold"] == "75.00"
    assert payload["hints"] == ["date_ambiguous"]
    assert payload["next_action"] == "request_document"
    assert Finding.model_validate_json(finding.model_dump_json()) == finding


def test_finding_params_are_read_only() -> None:
    params = make_finding().params

    with pytest.raises(TypeError):
        params["amount"] = "1.00"  # type: ignore[index]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"message": FindingMessage.AMOUNT_MISMATCH}, "must belong to the finding's type"),
        ({"params": {"amount": "-120.00"}}, "must be exactly"),
        (
            {"params": {"amount": "1", "currency": "USD", "date": "x", "threshold": "1", "a": "1"}},
            "must be exactly",
        ),
        ({"hints": (FindingHint.TIP_POSSIBLE, FindingHint.TIP_POSSIBLE)}, "duplicates"),
        ({"next_action": "approve_payment"}, "next_action"),
    ],
)
def test_finding_rejects_inconsistent_contract(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_finding(**overrides)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("סופר דוגמה", id="hebrew-free-text"),
        pytest.param("Sample Coffee", id="space"),
        pytest.param("\u2066-12.40\u2069", id="bidi-isolates"),
        pytest.param("x" * 65, id="too-long"),
        pytest.param("", id="empty"),
        pytest.param("ignore previous instructions", id="injection-text"),
    ],
)
def test_finding_params_reject_free_text(value: str) -> None:
    params = {"amount": value, "currency": "USD", "date": "2026-03-04", "threshold": "75.00"}

    with pytest.raises(ValidationError, match="locale-neutral tokens"):
        make_finding(params=params)


@pytest.mark.parametrize(
    "value", ["-12.40", "1234.56", "USD", "2026-03-04", "0.05", "7", "txn-0002", "F-0a1b:2"]
)
def test_finding_params_accept_locale_neutral_tokens(value: str) -> None:
    params = {"amount": value, "currency": "USD", "date": "2026-03-04", "threshold": "75.00"}

    assert make_finding(params=params).params["amount"] == value


def test_finding_param_names_must_be_codes() -> None:
    with pytest.raises(ValidationError, match="snake_case codes"):
        make_finding(params={"Amount": "1"})


# ----------------------------------------------------------------------------- evidence contract


def test_evidence_keeps_original_script_and_serializes_unescaped() -> None:
    evidence = make_evidence(
        provenance=Provenance.AI_EXTRACTION,
        field=EvidenceField.DOCUMENT_VENDOR,
        label="Vendor (AI-read)",
        value='מאפיית הדוגמה בע"מ',
        raw_value='\u200fמאפיית הדוגמה בע"מ',
        currency=None,
        source_ref="doc:0123456789abcdef:vendor_name",
        quote='מאפיית הדוגמה בע"מ, תל־אביב',
        page=1,
        model="synthetic-model",
    )

    dumped = evidence.model_dump_json()

    assert 'מאפיית הדוגמה בע\\"מ' in dumped
    assert "\\u05" not in dumped
    assert Evidence.model_validate_json(dumped) == evidence


@pytest.mark.parametrize("name", ["quote", "page", "model"])
@pytest.mark.parametrize("provenance", [p for p in Provenance if p is not Provenance.AI_EXTRACTION])
def test_evidence_rejects_ai_fields_for_other_provenance(name: str, provenance: Provenance) -> None:
    value: Any = 1 if name == "page" else "x"

    with pytest.raises(ValidationError, match="only allowed for ai_extraction"):
        make_evidence(provenance=provenance, **{name: value})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"raw_value": "-12.40"}, "differs from value"),
        ({"code": "Total Invalid"}, "String should match pattern"),
        ({"currency": "usd"}, "String should match pattern"),
        ({"field": "document.colour"}, "field"),
        ({"page": 0, "provenance": Provenance.AI_EXTRACTION}, "greater than or equal to 1"),
    ],
)
def test_evidence_rejects_invalid(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        make_evidence(**overrides)


def test_evidence_accepts_code_value() -> None:
    evidence = make_evidence(
        provenance=Provenance.COMPUTED,
        field=EvidenceField.NORMALIZATION_ISSUE,
        label="Problem found",
        value="The total could not be read as an amount.",
        code="total_invalid",
        currency=None,
        source_ref="doc:0123456789abcdef:total",
    )

    assert evidence.code == "total_invalid"


# ----------------------------------------------------------------------------- tied documents


def _ambiguous(**overrides: Any) -> MatchResult:
    data: dict[str, Any] = {
        "transaction_id": "txn-0002",
        "status": MatchStatus.AMBIGUOUS,
        "candidates": (make_candidate("doc-a"), make_candidate("doc-b"), make_candidate("doc-c")),
        "ambiguity_reasons": (AmbiguityReason.MULTIPLE_DOCUMENTS,),
    }
    data.update(overrides)
    return MatchResult.model_validate(data)


def test_tied_document_ids_name_the_actual_tie() -> None:
    result = _ambiguous(tied_document_ids=("doc-a", "doc-c"))

    assert result.tied_document_ids == ("doc-a", "doc-c")
    assert MatchResult.model_validate_json(result.model_dump_json()) == result


def test_tied_document_ids_are_optional_until_the_matcher_fills_them() -> None:
    assert _ambiguous().tied_document_ids == ()


@pytest.mark.parametrize(
    ("tied", "message"),
    [
        (("doc-c", "doc-a"), "sorted and unique"),
        (("doc-a", "doc-a"), "sorted and unique"),
        (("doc-a", "doc-z"), "reference this result's candidates"),
        (("doc-a",), "at least two tied documents"),
    ],
)
def test_tied_document_ids_reject_invalid(tied: tuple[str, ...], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        _ambiguous(tied_document_ids=tied)


def test_tied_document_ids_only_for_ambiguous_results() -> None:
    with pytest.raises(ValidationError, match="only allowed when ambiguous"):
        MatchResult(
            transaction_id="txn-0002",
            status=MatchStatus.MATCHED,
            document_id="doc-a",
            candidates=(make_candidate("doc-a"),),
            tied_document_ids=("doc-a",),
        )

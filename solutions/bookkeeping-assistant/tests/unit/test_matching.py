import random
from datetime import date, timedelta
from decimal import Decimal, localcontext

import pytest

from bookkeeping_assistant.config import ReviewConfig, ScoreWeights
from bookkeeping_assistant.matching import (
    MAX_COMPETING_CANDIDATES,
    DocumentExclusion,
    ExcludedDocument,
    MatchingOutcome,
    match_transactions,
    vendor_similarity,
)
from bookkeeping_assistant.models import (
    AmbiguityReason,
    BankTransaction,
    DocumentType,
    MatchAssumption,
    MatchResult,
    MatchStatus,
    NormalizationIssue,
    NormalizedDocument,
)
from bookkeeping_assistant.normalization import normalize_vendor

CONFIG = ReviewConfig()
DAY = date(2026, 3, 10)


def txn(
    number: int,
    amount: str = "-12.40",
    description: str = "SQ *SAMPLE COFFEE",
    posted: date = DAY,
) -> BankTransaction:
    return BankTransaction(
        id=f"t{number}",
        row_number=number + 1,
        posted_date=posted,
        description=description,
        amount=Decimal(amount),
        raw={"date": posted.isoformat(), "description": description, "amount": amount},
    )


def doc(
    name: str,
    total: str | None = "12.40",
    vendor: str | None = "Sample Coffee Co.",
    on: date | None = DAY,
    document_type: DocumentType = DocumentType.RECEIPT,
    currency: str | None = "USD",
    issues: tuple[NormalizationIssue, ...] = (),
) -> NormalizedDocument:
    return NormalizedDocument(
        document_id=name,
        document_type=document_type,
        vendor=vendor,
        vendor_normalized=(normalize_vendor(vendor) or None) if vendor else None,
        document_date=on,
        total=None if total is None else Decimal(total),
        currency=currency,
        issues=issues,
    )


def days(n: int) -> date:
    """A document date n days before the posted date (negative: after it)."""
    return DAY - timedelta(days=n)


def match(
    transactions: list[BankTransaction],
    documents: list[NormalizedDocument],
    config: ReviewConfig = CONFIG,
) -> dict[str, MatchResult]:
    outcome = match_transactions(transactions, documents, config)
    return {result.transaction_id: result for result in outcome.results}


def scores(result: MatchResult) -> dict[str, float]:
    return {c.document_id: c.breakdown.total for c in result.candidates}


# ============================================================================= scoring


def test_exact_match_has_full_breakdown() -> None:
    result = match([txn(1)], [doc("d1", on=days(1))])["t1"]

    assert result.status is MatchStatus.MATCHED
    assert result.document_id == "d1"
    (candidate,) = result.candidates
    assert candidate.assumptions == ()
    assert candidate.breakdown.model_dump() == {
        "amount_delta": Decimal("0.00"),
        "amount_component": 0.5,
        "date_delta_days": 1,
        "date_component": 0.16,  # 0.2 x (1 - 1/5)
        "vendor_similarity": 1.0,
        "vendor_component": 0.3,
        "total": 0.96,
    }


@pytest.mark.parametrize(
    ("amount", "total", "component", "delta"),
    [
        ("-12.40", "12.40", 0.5, "0.00"),
        ("-12.90", "12.40", 0.25, "0.50"),  # within 5% of the document total
        ("-21.00", "20.00", 0.25, "1.00"),  # exactly 5%: still close
        ("-21.01", "20.00", 0.0, "1.01"),  # just over 5%
        ("-19.00", "20.00", 0.25, "-1.00"),  # outflow below the total
        ("-14.88", "12.40", 0.0, "2.48"),  # a tip added after the receipt printed
    ],
)
def test_amount_component(amount: str, total: str, component: float, delta: str) -> None:
    result = match([txn(1, amount)], [doc("d1", total)])["t1"]

    assert result.status is MatchStatus.MATCHED  # the vendor match keeps it a candidate
    breakdown = result.candidates[0].breakdown
    assert breakdown.amount_component == component
    assert breakdown.amount_delta == Decimal(delta)


def test_score_uses_configured_weights() -> None:
    weights = ScoreWeights(amount_exact=0.6, amount_close=0.3, date=0.1, vendor=0.3)
    config = ReviewConfig(weights=weights)

    breakdown = match([txn(1)], [doc("d1")], config)["t1"].candidates[0].breakdown

    assert (breakdown.amount_component, breakdown.date_component) == (0.6, 0.1)
    assert (breakdown.vendor_component, breakdown.total) == (0.3, 1.0)


def test_perfect_match_with_weights_whose_rounded_parts_exceed_one() -> None:
    """Codex reproduction: each weight rounds up at 6 places, but the weights sum to 1.0."""
    weights = ScoreWeights(
        amount_exact=0.3333336, amount_close=0.25, date=0.3333336, vendor=0.3333328
    )

    result = match([txn(1)], [doc("d1")], ReviewConfig(weights=weights))["t1"]

    assert (result.status, result.document_id) == (MatchStatus.MATCHED, "d1")
    assert result.candidates[0].breakdown.total == 1.0


def assert_consistent(outcome: MatchingOutcome) -> None:
    """Every total is the rounded sum of its unrounded parts: within the rounding of 4 values."""
    for result in outcome.results:
        for candidate in (*result.candidates, *result.competing_candidates):
            b = candidate.breakdown
            parts = b.amount_component + b.date_component + b.vendor_component
            assert abs(parts - b.total) <= 4 * 0.5e-6
            assert 0.0 <= b.total <= 1.0


@pytest.mark.parametrize(
    "weights",
    [
        ScoreWeights(),
        ScoreWeights(amount_exact=0.3333336, amount_close=0.25, date=0.3333336, vendor=0.3333328),
        ScoreWeights(amount_exact=0.33, amount_close=0.2, date=0.56, vendor=0.11),
        ScoreWeights(amount_exact=0.1234567, amount_close=0.0999999, date=0.4, vendor=0.4765433),
    ],
)
def test_score_breakdowns_are_internally_consistent(weights: ScoreWeights) -> None:
    outcome = match_transactions(*scenario(), ReviewConfig(weights=weights))

    assert_consistent(outcome)


# ============================================================================= candidates


def test_no_candidate_without_exact_amount_or_similar_vendor() -> None:
    result = match([txn(1, "-12.90")], [doc("d1", vendor="Office Supply Depot")])["t1"]

    assert result == MatchResult(transaction_id="t1", status=MatchStatus.UNMATCHED)


def test_conflicting_merchants_with_exact_amount_stay_unmatched_with_candidate() -> None:
    """Codex reproduction: Walmart -$80 and a Target $80 receipt on the same day."""
    result = match([txn(1, "-80.00", "WALMART")], [doc("d1", "80.00", "Target")])["t1"]

    assert result.status is MatchStatus.UNMATCHED
    assert result.document_id is None
    (candidate,) = result.candidates  # kept as evidence for review
    breakdown = candidate.breakdown
    assert (breakdown.amount_delta, breakdown.amount_component) == (Decimal("0.00"), 0.5)
    assert breakdown.vendor_similarity < CONFIG.vendor_min_similarity
    assert candidate.assumptions == ()


def test_conflicting_merchant_does_not_take_a_document_from_the_right_merchant() -> None:
    transactions = [txn(1, "-80.00", "WALMART"), txn(2, "-80.00", "TARGET #0412")]

    results = match(transactions, [doc("d1", "80.00", "Target")])

    assert (results["t2"].status, results["t2"].document_id) == (MatchStatus.MATCHED, "d1")
    assert results["t1"].status is MatchStatus.UNMATCHED
    assert [c.document_id for c in results["t1"].candidates] == ["d1"]


def test_conflicting_merchant_does_not_make_a_real_match_ambiguous() -> None:
    """The Target receipt (0.84) is within the margin of the Walmart one (0.88), yet unrelated."""
    walmart = txn(1, "-80.00", "WALMART")
    documents = [doc("d1", "80.00", "Walmart", on=days(3)), doc("d2", "80.00", "Target")]

    result = match([walmart], documents)["t1"]

    assert (result.status, result.document_id) == (MatchStatus.MATCHED, "d1")
    assert [c.document_id for c in result.candidates] == ["d1", "d2"]


@pytest.mark.parametrize(
    "description",
    [
        "SQ *SAMPLE COFFEE",
        "POS DEBIT SQ *SAMPLE COFFEE 03/09 CARD 1234",
        "PURCHASE AUTHORIZED ON 03/09 SQ *SAMPLE COFFEE",
        "TST* SAMPLE COFFEE #0412",
        "SQ *SAMPLE COFFE",  # truncated by the bank
    ],
)
def test_noisy_bank_descriptions_still_match_on_exact_amount(description: str) -> None:
    result = match([txn(1, description=description)], [doc("d1")])["t1"]

    assert (result.status, result.document_id) == (MatchStatus.MATCHED, "d1")


@pytest.mark.parametrize(
    ("description", "vendor"),
    [
        ("SQ *SAMPLE COFFEE", None),  # the document names no merchant
        ("***", "Target"),  # the bank description has no usable merchant
    ],
)
def test_unknown_merchant_is_not_a_conflict(description: str, vendor: str | None) -> None:
    result = match([txn(1, description=description)], [doc("d1", vendor=vendor)])["t1"]

    assert (result.status, result.document_id) == (MatchStatus.MATCHED, "d1")
    assert result.candidates[0].breakdown.vendor_similarity == 0.0


def test_missing_vendor_matches_only_on_exact_amount() -> None:
    exact = match([txn(1)], [doc("d1", vendor=None)])["t1"]
    close = match([txn(1, "-12.90")], [doc("d1", vendor=None)])["t1"]

    assert exact.status is MatchStatus.MATCHED
    assert exact.candidates[0].breakdown.vendor_similarity == 0.0
    assert close.status is MatchStatus.UNMATCHED


@pytest.mark.parametrize(
    ("document_type", "days_before", "is_candidate"),
    [
        (DocumentType.RECEIPT, 5, True),
        (DocumentType.RECEIPT, 6, False),
        (DocumentType.RECEIPT, -2, True),  # document dated after the posting
        (DocumentType.RECEIPT, -3, False),
        (DocumentType.INVOICE, 60, True),
        (DocumentType.INVOICE, 61, False),
        (DocumentType.INVOICE, -2, True),
        (DocumentType.INVOICE, -3, False),
        (DocumentType.OTHER, 5, True),  # unrecognized types use the narrower receipt window
        (DocumentType.OTHER, 6, False),
    ],
)
def test_date_windows(document_type: DocumentType, days_before: int, is_candidate: bool) -> None:
    document = doc("d1", on=days(days_before), document_type=document_type)

    result = match([txn(1)], [document])["t1"]

    assert bool(result.candidates) is is_candidate


@pytest.mark.parametrize(
    ("days_before", "component"),
    [(0, 0.2), (1, 0.16), (5, 0.0), (-1, 0.1), (-2, 0.0)],
)
def test_date_component_scales_with_the_window_side(days_before: int, component: float) -> None:
    result = match([txn(1)], [doc("d1", on=days(days_before))])["t1"]

    breakdown = result.candidates[0].breakdown
    assert breakdown.date_delta_days == days_before
    assert breakdown.date_component == component


def test_zero_width_window_accepts_only_the_same_day() -> None:
    config = ReviewConfig(receipt_days_before=0, receipt_days_after=0)

    same_day = match([txn(1)], [doc("d1")], config)["t1"]
    day_before = match([txn(1)], [doc("d1", on=days(1))], config)["t1"]
    day_after = match([txn(1)], [doc("d1", on=days(-1))], config)["t1"]

    assert same_day.candidates[0].breakdown.date_component == 0.2  # no division by zero
    assert day_before.status is day_after.status is MatchStatus.UNMATCHED


def test_vendor_similarity_tolerates_bank_description_noise() -> None:
    noisy = txn(1, "-14.88", description="POS DEBIT SQ *SAMPLE COFFEE 03/09 CARD 1234")

    result = match([noisy], [doc("d1")])["t1"]

    assert result.status is MatchStatus.MATCHED
    assert result.candidates[0].breakdown.vendor_similarity == 1.0


@pytest.mark.parametrize(
    ("document_key", "transaction_key", "low", "high"),
    [
        ("SAMPLE COFFEE", "SAMPLE COFFEE", 1.0, 1.0),
        ("SAMPLE COFFEE", "POS DEBIT SAMPLE COFFEE 03 09", 1.0, 1.0),
        ("SAMPLE COFFEE", "SAMPLE COFFE", 0.9, 0.99),  # truncated by the bank
        ("SAMPLE COFFEE", "SAMPLE BAKERY", 0.5, 0.75),
        ("STUDIO 54", "STUDIO 55", 0.5, 0.99),  # similar, but never identical
        ("OFFICE SUPPLY DEPOT", "SAMPLE COFFEE", 0.0, 0.4),
        ("", "SAMPLE COFFEE", 0.0, 0.0),
        (None, "SAMPLE COFFEE", 0.0, 0.0),
        ("SAMPLE COFFEE", "", 0.0, 0.0),
    ],
)
def test_vendor_similarity(
    document_key: str | None, transaction_key: str, low: float, high: float
) -> None:
    assert low <= vendor_similarity(document_key, transaction_key) <= high


def test_vendor_similarity_bounds_long_keys() -> None:
    long_key = "A " * 5000

    assert 0.0 <= vendor_similarity(long_key, long_key[::-1] + "B") <= 1.0


# ============================================================================= eligibility


def test_only_outflows_are_matched() -> None:
    outcome = match_transactions(
        [txn(1, "12.40"), txn(2, "0.00"), txn(3, "-12.40")], [doc("d1")], CONFIG
    )

    assert [r.transaction_id for r in outcome.results] == ["t3"]


@pytest.mark.parametrize(
    ("document", "reasons"),
    [
        (doc("d1", document_type=DocumentType.UNREADABLE), (DocumentExclusion.UNREADABLE,)),
        (doc("d1", total=None), (DocumentExclusion.TOTAL_UNAVAILABLE,)),
        (doc("d1", total="0.00"), (DocumentExclusion.TOTAL_NOT_POSITIVE,)),
        (doc("d1", total="-12.40"), (DocumentExclusion.TOTAL_NOT_POSITIVE,)),
        (doc("d1", on=None), (DocumentExclusion.DATE_UNAVAILABLE,)),
        (doc("d1", currency="EUR"), (DocumentExclusion.CURRENCY_NOT_USD,)),
        (doc("d1", currency="CAD"), (DocumentExclusion.CURRENCY_NOT_USD,)),
        (
            doc("d1", total=None, on=None, document_type=DocumentType.UNREADABLE),
            (
                DocumentExclusion.UNREADABLE,
                DocumentExclusion.TOTAL_UNAVAILABLE,
                DocumentExclusion.DATE_UNAVAILABLE,
            ),
        ),
    ],
)
def test_ineligible_documents_are_excluded_with_reasons(
    document: NormalizedDocument, reasons: tuple[DocumentExclusion, ...]
) -> None:
    outcome = match_transactions([txn(1)], [document], CONFIG)

    assert outcome.excluded_documents == (ExcludedDocument(document_id="d1", reasons=reasons),)
    assert outcome.results[0].status is MatchStatus.UNMATCHED
    assert outcome.results[0].candidates == ()


# ============================================================================= uncertainty


@pytest.mark.parametrize(
    ("currency", "issues", "assumptions"),
    [
        (None, (NormalizationIssue.CURRENCY_AMBIGUOUS,), (MatchAssumption.CURRENCY_UNCONFIRMED,)),
        (None, (NormalizationIssue.CURRENCY_MISSING,), (MatchAssumption.CURRENCY_UNCONFIRMED,)),
        ("USD", (NormalizationIssue.DATE_AMBIGUOUS,), (MatchAssumption.DATE_AMBIGUOUS,)),
        (
            None,
            (NormalizationIssue.DATE_AMBIGUOUS, NormalizationIssue.CURRENCY_AMBIGUOUS),
            (MatchAssumption.CURRENCY_UNCONFIRMED, MatchAssumption.DATE_AMBIGUOUS),
        ),
    ],
)
def test_uncertain_document_data_is_recorded_as_an_assumption(
    currency: str | None,
    issues: tuple[NormalizationIssue, ...],
    assumptions: tuple[MatchAssumption, ...],
) -> None:
    result = match([txn(1)], [doc("d1", currency=currency, issues=issues)])["t1"]

    assert result.status is MatchStatus.MATCHED
    assert result.candidates[0].assumptions == assumptions
    assert result.model_dump(mode="json")["candidates"][0]["assumptions"] == [
        a.value for a in assumptions
    ]


def test_uncertain_and_confirmed_documents_with_equal_scores_are_ambiguous() -> None:
    """Never silently prefer one document just because the other is uncertain, or vice versa."""
    documents = [doc("d1"), doc("d2", currency=None, issues=(NormalizationIssue.CURRENCY_MISSING,))]

    result = match([txn(1)], documents)["t1"]

    assert result.status is MatchStatus.AMBIGUOUS
    assert result.ambiguity_reasons == (AmbiguityReason.MULTIPLE_DOCUMENTS,)


# ============================================================================= competition


def test_best_of_several_documents_wins_and_others_stay_listed() -> None:
    result = match([txn(1)], [doc("d2", on=days(3)), doc("d1")])["t1"]

    assert (result.status, result.document_id) == (MatchStatus.MATCHED, "d1")
    assert scores(result) == {"d1": 1.0, "d2": 0.88}
    assert [c.document_id for c in result.candidates] == ["d1", "d2"]  # best first


def test_near_tied_documents_make_the_transaction_ambiguous() -> None:
    """Two indistinguishable receipts (e.g. a re-scan) for one transaction."""
    result = match([txn(1)], [doc("d1"), doc("d2")])["t1"]

    assert result.status is MatchStatus.AMBIGUOUS
    assert result.document_id is None
    assert result.ambiguity_reasons == (AmbiguityReason.MULTIPLE_DOCUMENTS,)
    assert scores(result) == {"d1": 1.0, "d2": 1.0}
    assert result.competing_candidates == ()


def test_near_tied_transactions_make_the_document_shared() -> None:
    """Two identical charges (a possible double charge) and one receipt."""
    results = match([txn(1), txn(2)], [doc("d1")])

    for own, other in (("t1", "t2"), ("t2", "t1")):
        result = results[own]
        assert result.status is MatchStatus.AMBIGUOUS
        assert result.ambiguity_reasons == (AmbiguityReason.SHARED_DOCUMENT,)
        assert [(c.transaction_id, c.document_id) for c in result.competing_candidates] == [
            (other, "d1")
        ]


def test_clearly_weaker_transaction_loses_the_document_but_keeps_its_candidate() -> None:
    results = match([txn(1), txn(2, posted=DAY + timedelta(days=4))], [doc("d1")])

    assert (results["t1"].status, results["t1"].document_id) == (MatchStatus.MATCHED, "d1")
    loser = results["t2"]
    assert loser.status is MatchStatus.UNMATCHED
    assert scores(loser) == {"d1": 0.84}  # preserved as evidence for review


def test_losing_transaction_falls_back_to_its_next_best_document() -> None:
    documents = [doc("d1"), doc("d2", total="12.00", on=DAY + timedelta(days=2))]
    transactions = [txn(1), txn(2, posted=DAY + timedelta(days=2))]

    results = match(transactions, documents)

    assert (results["t1"].status, results["t1"].document_id) == (MatchStatus.MATCHED, "d1")
    assert (results["t2"].status, results["t2"].document_id) == (MatchStatus.MATCHED, "d2")
    assert scores(results["t2"]) == {"d1": 0.92, "d2": 0.75}


def test_documents_in_a_near_tie_are_held_back_from_weaker_transactions() -> None:
    """t1 cannot choose between d1 and d2, so neither is handed to the clearly weaker t2."""
    transactions = [txn(1), txn(2, posted=DAY + timedelta(days=4))]

    results = match(transactions, [doc("d1"), doc("d2")])

    assert results["t1"].status is MatchStatus.AMBIGUOUS
    assert results["t2"].status is MatchStatus.UNMATCHED
    assert scores(results["t2"]) == {"d1": 0.84, "d2": 0.84}


def test_connected_near_ties_are_ambiguous_together() -> None:
    """t1 is torn between d1 and d2, and t2 is near-tied with t1 for d2."""
    transactions = [txn(1), txn(2, posted=DAY - timedelta(days=1))]
    documents = [doc("d1"), doc("d2", on=days(1))]

    results = match(transactions, documents)

    assert results["t1"].ambiguity_reasons == (
        AmbiguityReason.MULTIPLE_DOCUMENTS,
        AmbiguityReason.SHARED_DOCUMENT,
    )
    assert [c.transaction_id for c in results["t1"].competing_candidates] == ["t2"]
    assert results["t2"].ambiguity_reasons == (AmbiguityReason.SHARED_DOCUMENT,)
    assert [c.document_id for c in results["t2"].competing_candidates] == ["d2"]


def test_recurring_purchases_a_week_apart_match_distinctly() -> None:
    week = timedelta(days=7)
    transactions = [txn(1), txn(2, posted=DAY + week)]
    documents = [doc("d1"), doc("d2", on=DAY + week)]

    results = match(transactions, documents)

    assert [results[t].document_id for t in ("t1", "t2")] == ["d1", "d2"]
    assert all(r.status is MatchStatus.MATCHED for r in results.values())


def test_recurring_purchases_on_adjacent_days_are_ambiguous_unless_the_margin_allows() -> None:
    """One day of date difference is worth 0.04, inside the default 0.05 margin."""
    transactions = [txn(1), txn(2, posted=DAY + timedelta(days=1))]
    documents = [doc("d1"), doc("d2", on=DAY + timedelta(days=1))]

    default = match(transactions, documents)
    tighter = match(transactions, documents, ReviewConfig(ambiguity_margin=0.03))

    assert all(r.status is MatchStatus.AMBIGUOUS for r in default.values())
    assert [tighter[t].document_id for t in ("t1", "t2")] == ["d1", "d2"]


@pytest.mark.parametrize(
    ("margin", "status"),
    [
        (0.04, MatchStatus.AMBIGUOUS),  # scores 1.0 and 0.96: a 0.04 gap is within the margin
        (0.039, MatchStatus.MATCHED),
    ],
)
def test_ambiguity_margin_is_inclusive(margin: float, status: MatchStatus) -> None:
    config = ReviewConfig(ambiguity_margin=margin)

    result = match([txn(1)], [doc("d1"), doc("d2", on=days(1))], config)["t1"]

    assert result.status is status


def test_ambiguity_margin_tolerates_float_rounding() -> None:
    """Scores 0.9 and 0.84 differ by exactly 0.06, but 0.9 - 0.06 is 0.8400000000000001."""
    config = ReviewConfig(ambiguity_margin=0.06)

    result = match([txn(1)], [doc("d1", on=days(-1)), doc("d2", on=days(4))], config)["t1"]

    assert scores(result) == {"d1": 0.9, "d2": 0.84}
    assert result.status is MatchStatus.AMBIGUOUS


def test_competing_candidates_are_capped_but_every_competitor_has_its_own_result() -> None:
    transactions = [txn(n) for n in range(1, 31)]  # 30 identical charges, 2 identical receipts

    results = match(transactions, [doc("d1"), doc("d2")])

    assert len(results) == 30
    for result in results.values():
        assert result.status is MatchStatus.AMBIGUOUS
        assert result.ambiguity_reasons == (
            AmbiguityReason.MULTIPLE_DOCUMENTS,
            AmbiguityReason.SHARED_DOCUMENT,
        )
        assert len(result.competing_candidates) == MAX_COMPETING_CANDIDATES
        assert {c.document_id for c in result.candidates} == {"d1", "d2"}
    # Ties break by transaction ID, so t1's competitors are the next IDs in order.
    assert [(c.transaction_id, c.document_id) for c in results["t1"].competing_candidates][:4] == [
        ("t10", "d1"),
        ("t10", "d2"),
        ("t11", "d1"),
        ("t11", "d2"),
    ]


def test_identical_scores_are_ambiguous_even_with_zero_margin() -> None:
    config = ReviewConfig(ambiguity_margin=0.0)

    result = match([txn(1)], [doc("d1"), doc("d2")], config)["t1"]

    assert result.status is MatchStatus.AMBIGUOUS


# ============================================================================= determinism


def scenario() -> tuple[list[BankTransaction], list[NormalizedDocument]]:
    transactions = [
        txn(1),
        txn(2),  # double charge with t1
        txn(3, "-245.99", "OFFICE SUPPLY DEPOT #12", DAY + timedelta(days=1)),
        txn(4, "-14.88", "POS DEBIT TST* SAMPLE BISTRO", DAY + timedelta(days=3)),
        txn(5, "-99.00", "UNKNOWN MERCHANT"),
        txn(6, "500.00", "DEPOSIT"),
        txn(7, "-30.00", "SQ *STUDIO 54", DAY + timedelta(days=20)),
        txn(8, "-30.00", "SQ *STUDIO 54", DAY + timedelta(days=21)),
    ]
    documents = [
        doc("d1"),
        doc(
            "d2",
            "245.99",
            "Office Supply Depot, Inc.",
            days(20),
            DocumentType.INVOICE,
            currency=None,
            issues=(NormalizationIssue.CURRENCY_MISSING,),
        ),
        doc("d3", "12.40", "Sample Bistro", DAY + timedelta(days=2)),
        doc("d4", "30.00", "Studio 54", DAY + timedelta(days=20)),
        doc("d5", "30.00", "Studio 54", DAY + timedelta(days=21)),
        doc("d6", "18.50", "Café Exemple", currency="EUR"),
        doc("d7", None, None, None, DocumentType.UNREADABLE),
    ]
    return transactions, documents


def test_scenario_outcome() -> None:
    results = match(*scenario())

    statuses = {t: (r.status.value, r.document_id) for t, r in results.items()}
    assert statuses == {
        "t1": ("ambiguous", None),
        "t2": ("ambiguous", None),
        "t3": ("matched", "d2"),
        "t4": ("matched", "d3"),
        "t5": ("unmatched", None),
        "t7": ("ambiguous", None),
        "t8": ("ambiguous", None),
    }
    assert results["t3"].candidates[0].assumptions == (MatchAssumption.CURRENCY_UNCONFIRMED,)


@pytest.mark.parametrize("seed", range(20))
def test_outcome_does_not_depend_on_input_order(seed: int) -> None:
    transactions, documents = scenario()
    expected = match_transactions(transactions, documents, CONFIG)

    rng = random.Random(seed)
    rng.shuffle(transactions)
    rng.shuffle(documents)
    shuffled = match_transactions(transactions, documents, CONFIG)

    assert shuffled == expected
    assert shuffled.model_dump_json() == expected.model_dump_json()


def test_results_follow_csv_row_order_and_round_trip() -> None:
    outcome = match_transactions(*scenario(), CONFIG)

    assert [r.transaction_id for r in outcome.results] == [
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t7",
        "t8",
    ]
    assert [d.document_id for d in outcome.excluded_documents] == ["d6", "d7"]
    assert MatchingOutcome.model_validate_json(outcome.model_dump_json()) == outcome


def test_outcome_does_not_depend_on_the_callers_decimal_context() -> None:
    expected = match_transactions(*scenario(), CONFIG)

    with localcontext() as context:
        context.prec = 2
        assert match_transactions(*scenario(), CONFIG) == expected


# ============================================================================= input validation


@pytest.mark.parametrize(
    ("transactions", "documents", "message"),
    [
        ([txn(1), txn(1)], [doc("d1")], "duplicate transaction id: t1"),
        ([txn(1)], [doc("d1"), doc("d1")], "duplicate document id: d1"),
    ],
)
def test_duplicate_ids_are_rejected(
    transactions: list[BankTransaction], documents: list[NormalizedDocument], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        match_transactions(transactions, documents, CONFIG)


def test_empty_inputs() -> None:
    assert match_transactions([], [], CONFIG) == MatchingOutcome(results=(), excluded_documents=())
    assert match([txn(1)], []) == {
        "t1": MatchResult(transaction_id="t1", status=MatchStatus.UNMATCHED)
    }

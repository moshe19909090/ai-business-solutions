"""Stage 4: deterministic matching of bank transactions to normalized documents.

Only outflow transactions are matched, and only against documents with a positive total, a
date, and a currency that is US dollars or unknown. Every decision is plain arithmetic on the
inputs and the `ReviewConfig`, so the same inputs always give the same outcome, whatever their
order.

Scoring (weights from `ReviewConfig.weights`, recorded in each `ScoreBreakdown`):
- amount: `amount_exact` if the outflow equals the document total, `amount_close` if it is
  within `amount_close_ratio` of the total, else 0
- date: `date` x (1 - |days| / window side), where the side is the window before or after the
  posted date that the document date falls on. Documents outside the window are not candidates.
- vendor: `vendor` x `vendor_similarity`

A pair is a candidate only if its dates fit the window and the amount is exact or the vendors
are at least `vendor_min_similarity` alike. Components are rounded to `SCORE_DECIMALS` for
display; the total is the rounded sum of the unrounded components.

An exact amount with conflicting merchants (both known, but less than `vendor_min_similarity`
alike, e.g. Walmart vs Target) stays a candidate as review evidence, but is never assigned and
never makes another match ambiguous. A merchant missing on either side is unknown, not
conflicting, so the exact amount can still match.

Assignment is greedy and conservative. The best remaining pair is assigned one-to-one only when
nothing else is within `ambiguity_margin` of it: no other document for that transaction and no
other transaction for that document. Otherwise every near-tied pair connected to it is marked
ambiguous, and those documents are held back rather than guessed. Greedy assignment is not
globally optimal. Near ties are surfaced instead of resolved.

Uncertain document data never silently becomes a confirmed match. A candidate relying on an
unknown currency or an ambiguous date lists that in `MatchCandidate.assumptions`.
"""

from collections import deque
from collections.abc import Iterable, Sequence
from decimal import localcontext
from difflib import SequenceMatcher
from enum import StrEnum
from functools import lru_cache
from typing import Final

from pydantic import Field

from bookkeeping_assistant.config import ReviewConfig
from bookkeeping_assistant.models import (
    SCORE_TOLERANCE,
    AmbiguityReason,
    BankTransaction,
    Direction,
    DocumentType,
    DomainModel,
    MatchAssumption,
    MatchCandidate,
    MatchResult,
    MatchStatus,
    NonEmptyStr,
    NormalizationIssue,
    NormalizedDocument,
    ScoreBreakdown,
)
from bookkeeping_assistant.money import MONEY_CONTEXT
from bookkeeping_assistant.normalization import SUPPORTED_CURRENCY, normalize_vendor

SCORE_DECIMALS: Final = 6
"""Scores are rounded so that equal inputs compare equal and JSON stays readable."""
MAX_VENDOR_KEY_LENGTH: Final = 100
"""Vendor keys are truncated before comparison, bounding `SequenceMatcher`'s quadratic cost."""
MAX_COMPETING_CANDIDATES: Final = 10
"""Cap on `MatchResult.competing_candidates` (highest scores first). It bounds report size when
many transactions contend for the same documents. Nothing is lost: every competing transaction
has its own ambiguous result listing the contested document among its candidates."""


class DocumentExclusion(StrEnum):
    """Why a document cannot be matched. Phase 5 turns these into reviewer findings."""

    UNREADABLE = "unreadable"
    TOTAL_UNAVAILABLE = "total_unavailable"
    TOTAL_NOT_POSITIVE = "total_not_positive"
    DATE_UNAVAILABLE = "date_unavailable"
    CURRENCY_NOT_USD = "currency_not_usd"
    """The document states another currency, so its total cannot be compared with USD."""


class ExcludedDocument(DomainModel):
    document_id: NonEmptyStr
    reasons: tuple[DocumentExclusion, ...] = Field(min_length=1)


class MatchingOutcome(DomainModel):
    results: tuple[MatchResult, ...] = Field(
        description="One result per outflow transaction, in CSV row order."
    )
    excluded_documents: tuple[ExcludedDocument, ...] = Field(
        description="Documents that could not take part in matching, ordered by ID."
    )


# ----------------------------------------------------------------------------- vendor


def vendor_similarity(document_key: str | None, transaction_key: str | None) -> float:
    """How alike two normalized vendor keys are, in [0, 1]. Empty or missing keys score 0.

    The larger of two deterministic measures:
    - `difflib.SequenceMatcher` ratio of the keys (tolerates truncation and small spelling
      differences: "SAMPLE COFFE" vs "SAMPLE COFFEE")
    - the share of the document's words found in the transaction key (tolerates bank noise:
      "POS DEBIT SAMPLE COFFEE 03 02" contains every word of "SAMPLE COFFEE")
    """
    if not document_key or not transaction_key:
        return 0.0
    return _similarity(
        document_key[:MAX_VENDOR_KEY_LENGTH], transaction_key[:MAX_VENDOR_KEY_LENGTH]
    )


@lru_cache(maxsize=65536)
def _similarity(document_key: str, transaction_key: str) -> float:
    """Pure function of its arguments, so caching cannot change results."""
    ratio = SequenceMatcher(None, document_key, transaction_key, autojunk=False).ratio()
    document_words = set(document_key.split())
    shared = document_words & set(transaction_key.split())
    return round(max(ratio, len(shared) / len(document_words)), SCORE_DECIMALS)


# ----------------------------------------------------------------------------- matching


def match_transactions(
    transactions: Sequence[BankTransaction],
    documents: Sequence[NormalizedDocument],
    config: ReviewConfig,
) -> MatchingOutcome:
    """Match outflow transactions to documents. Inflows and zero amounts are not matched.

    Raises:
        ValueError: transaction IDs or document IDs are not unique.
    """
    _require_unique((t.id for t in transactions), "transaction")
    _require_unique((d.document_id for d in documents), "document")

    outflows = sorted(
        (t for t in transactions if t.direction is Direction.OUTFLOW),
        key=lambda t: (t.row_number, t.id),
    )
    eligible: list[NormalizedDocument] = []
    excluded: list[ExcludedDocument] = []
    for document in sorted(documents, key=lambda d: d.document_id):
        reasons = _exclusions(document)
        if reasons:
            excluded.append(ExcludedDocument(document_id=document.document_id, reasons=reasons))
        else:
            eligible.append(document)

    evidence: dict[str, list[MatchCandidate]] = {}  # every candidate, per transaction
    assignable: list[MatchCandidate] = []
    for transaction in outflows:
        transaction_key = normalize_vendor(transaction.description)
        for document in eligible:
            candidate = _score(transaction, transaction_key, document, config)
            if candidate is None:
                continue
            evidence.setdefault(transaction.id, []).append(candidate)
            if not _merchants_conflict(candidate, document, transaction_key, config):
                assignable.append(candidate)
    assignment = _Assignment(assignable, config.ambiguity_margin)
    return MatchingOutcome(
        results=tuple(
            assignment.result(t.id, tuple(sorted(evidence.get(t.id, ()), key=_score_order)))
            for t in outflows
        ),
        excluded_documents=tuple(excluded),
    )


def _require_unique(ids: Iterable[str], kind: str) -> None:
    seen: set[str] = set()
    for item in ids:
        if item in seen:
            raise ValueError(f"duplicate {kind} id: {item}")
        seen.add(item)


def _exclusions(document: NormalizedDocument) -> tuple[DocumentExclusion, ...]:
    reasons = []
    if document.document_type is DocumentType.UNREADABLE:
        reasons.append(DocumentExclusion.UNREADABLE)
    if document.total is None:
        reasons.append(DocumentExclusion.TOTAL_UNAVAILABLE)
    elif document.total <= 0:
        reasons.append(DocumentExclusion.TOTAL_NOT_POSITIVE)
    if document.document_date is None:
        reasons.append(DocumentExclusion.DATE_UNAVAILABLE)
    if document.currency is not None and document.currency != SUPPORTED_CURRENCY:
        reasons.append(DocumentExclusion.CURRENCY_NOT_USD)
    return tuple(reasons)


def _score(
    transaction: BankTransaction,
    transaction_key: str,
    document: NormalizedDocument,
    config: ReviewConfig,
) -> MatchCandidate | None:
    """Score one pair, or None if it is not a candidate. The document must be eligible."""
    if document.total is None or document.document_date is None:
        raise ValueError("document is not eligible for matching")
    weights = config.weights

    days_before, days_after = _date_window(document.document_type, config)
    delta_days = (transaction.posted_date - document.document_date).days
    if delta_days > days_before or -delta_days > days_after:
        return None
    side = days_before if delta_days >= 0 else days_after
    date_fraction = 1.0 if side == 0 else 1.0 - abs(delta_days) / side

    # A fixed context keeps the arithmetic independent of the caller's Decimal settings.
    with localcontext(MONEY_CONTEXT):
        amount_delta = transaction.amount.copy_negate() - document.total
        is_exact = amount_delta == 0
        is_close = amount_delta.copy_abs() <= config.amount_close_ratio * document.total
    if is_exact:
        amount_component = weights.amount_exact
    elif is_close:
        amount_component = weights.amount_close
    else:
        amount_component = 0.0

    similarity = vendor_similarity(document.vendor_normalized, transaction_key)
    if not is_exact and similarity < config.vendor_min_similarity:
        return None

    date_component = weights.date * date_fraction
    vendor_component = weights.vendor * similarity
    # The total comes from the unrounded components. Rounding each component first could push
    # a perfect score past 1.0 (e.g. three weights of about 1/3 that each round up).
    total = amount_component + date_component + vendor_component
    return MatchCandidate(
        transaction_id=transaction.id,
        document_id=document.document_id,
        breakdown=ScoreBreakdown(
            amount_delta=amount_delta,
            amount_component=_round(amount_component),
            date_delta_days=delta_days,
            date_component=_round(date_component),
            vendor_similarity=similarity,
            vendor_component=_round(vendor_component),
            total=_round(total),
        ),
        assumptions=_assumptions(document),
    )


def _merchants_conflict(
    candidate: MatchCandidate,
    document: NormalizedDocument,
    transaction_key: str,
    config: ReviewConfig,
) -> bool:
    """Both merchants are known and clearly different. Only exact-amount candidates can be
    in this state: other candidates already need `vendor_min_similarity`."""
    return (
        bool(document.vendor_normalized)
        and bool(transaction_key)
        and candidate.breakdown.vendor_similarity < config.vendor_min_similarity
    )


def _date_window(document_type: DocumentType, config: ReviewConfig) -> tuple[int, int]:
    """(days before, days after) the posted date. Unrecognized types use the narrower window."""
    if document_type is DocumentType.INVOICE:
        return config.invoice_days_before, config.invoice_days_after
    return config.receipt_days_before, config.receipt_days_after


def _assumptions(document: NormalizedDocument) -> tuple[MatchAssumption, ...]:
    assumptions = []
    if document.currency is None:
        assumptions.append(MatchAssumption.CURRENCY_UNCONFIRMED)
    if NormalizationIssue.DATE_AMBIGUOUS in document.issues:
        assumptions.append(MatchAssumption.DATE_AMBIGUOUS)
    return tuple(assumptions)


def _round(value: float) -> float:
    return round(value, SCORE_DECIMALS)


def _score_order(candidate: MatchCandidate) -> tuple[float, str, str]:
    return (-candidate.breakdown.total, candidate.transaction_id, candidate.document_id)


class _Assignment:
    """Greedy, conservative one-to-one assignment over scored candidates (single use).

    Repeatedly takes the best pair whose transaction and document are both still free, then
    collects every free pair connected to it through "near" pairs. A pair is near when its
    score is within the margin of the best free score of both its transaction and its document.
    A component that is just the one pair is a match. A larger component means a near tie: its
    transactions are all ambiguous, and its documents are held back from everyone.
    """

    def __init__(self, candidates: list[MatchCandidate], margin: float) -> None:
        self._margin = margin
        self._by_txn: dict[str, list[MatchCandidate]] = {}
        self._by_doc: dict[str, list[MatchCandidate]] = {}
        ranked = sorted(candidates, key=_score_order)
        for candidate in ranked:
            self._by_txn.setdefault(candidate.transaction_id, []).append(candidate)
            self._by_doc.setdefault(candidate.document_id, []).append(candidate)
        self._free_txns = set(self._by_txn)
        self._free_docs = set(self._by_doc)
        self._matched: dict[str, str] = {}
        # Ambiguous transaction -> (reasons, competing candidates)
        self._ambiguous: dict[str, tuple[list[AmbiguityReason], list[MatchCandidate]]] = {}
        for candidate in ranked:
            if self._is_free(candidate):
                self._resolve_from(candidate)

    def result(self, transaction_id: str, candidates: tuple[MatchCandidate, ...]) -> MatchResult:
        """`candidates` are all of the transaction's candidates, including unassignable ones."""
        if transaction_id in self._matched:
            return MatchResult(
                transaction_id=transaction_id,
                status=MatchStatus.MATCHED,
                document_id=self._matched[transaction_id],
                candidates=candidates,
            )
        if transaction_id in self._ambiguous:
            return self._ambiguous_result(transaction_id, candidates)
        # No candidates, every candidate document went to clearly better pairs, or the only
        # candidates have conflicting merchants.
        return MatchResult(
            transaction_id=transaction_id, status=MatchStatus.UNMATCHED, candidates=candidates
        )

    def _is_free(self, candidate: MatchCandidate) -> bool:
        return (
            candidate.transaction_id in self._free_txns and candidate.document_id in self._free_docs
        )

    def _resolve_from(self, top: MatchCandidate) -> None:
        """`top` is the best free pair: every free pair scores at most as much."""
        edges = self._near_component(top)
        txns = {edge.transaction_id for edge in edges}
        self._free_txns -= txns
        self._free_docs -= {edge.document_id for edge in edges}
        if len(edges) == 1:
            self._matched[top.transaction_id] = top.document_id
        else:
            self._record_ambiguity(edges)

    def _near_component(self, top: MatchCandidate) -> list[MatchCandidate]:
        # Free sets do not change while the component is collected, so bests can be cached.
        best_txn: dict[str, float] = {}
        best_doc: dict[str, float] = {}

        def is_near(candidate: MatchCandidate) -> bool:
            """Only called for free pairs, so both sides have a best free score."""
            txn, doc = candidate.transaction_id, candidate.document_id
            if txn not in best_txn:
                best_txn[txn] = self._best(self._by_txn[txn])
            if doc not in best_doc:
                best_doc[doc] = self._best(self._by_doc[doc])
            bar = max(best_txn[txn], best_doc[doc])
            return candidate.breakdown.total >= bar - self._margin - SCORE_TOLERANCE

        edges: list[MatchCandidate] = []
        seen_edges: set[tuple[str, str]] = set()
        seen_txns, seen_docs = {top.transaction_id}, {top.document_id}
        queue: deque[tuple[bool, str]] = deque(
            [(True, top.transaction_id), (False, top.document_id)]
        )
        while queue:
            is_txn, key = queue.popleft()
            for candidate in self._by_txn[key] if is_txn else self._by_doc[key]:
                pair = (candidate.transaction_id, candidate.document_id)
                if pair in seen_edges or not self._is_free(candidate) or not is_near(candidate):
                    continue
                seen_edges.add(pair)
                edges.append(candidate)
                if candidate.transaction_id not in seen_txns:
                    seen_txns.add(candidate.transaction_id)
                    queue.append((True, candidate.transaction_id))
                if candidate.document_id not in seen_docs:
                    seen_docs.add(candidate.document_id)
                    queue.append((False, candidate.document_id))
        return sorted(edges, key=_score_order)

    def _best(self, ranked: list[MatchCandidate]) -> float:
        """Best score among a ranked list's free pairs. The list must contain a free pair."""
        return next(c.breakdown.total for c in ranked if self._is_free(c))

    def _record_ambiguity(self, edges: list[MatchCandidate]) -> None:
        """Explain each transaction of a near-tied component. `edges` are in score order."""
        own_docs: dict[str, list[str]] = {}
        edges_by_doc: dict[str, list[MatchCandidate]] = {}
        for edge in edges:
            own_docs.setdefault(edge.transaction_id, []).append(edge.document_id)
            edges_by_doc.setdefault(edge.document_id, []).append(edge)
        for transaction_id, docs in own_docs.items():
            # Only the best few per document can make the capped list, even after excluding
            # this transaction's own edge, so large components stay cheap to explain.
            competitors = sorted(
                (
                    edge
                    for doc in docs
                    for edge in edges_by_doc[doc][: MAX_COMPETING_CANDIDATES + 1]
                    if edge.transaction_id != transaction_id
                ),
                key=_score_order,
            )[:MAX_COMPETING_CANDIDATES]
            reasons = []
            if len(docs) >= 2:
                reasons.append(AmbiguityReason.MULTIPLE_DOCUMENTS)
            if any(len(edges_by_doc[doc]) >= 2 for doc in docs):
                reasons.append(AmbiguityReason.SHARED_DOCUMENT)
            self._ambiguous[transaction_id] = (reasons, competitors)

    def _ambiguous_result(
        self, transaction_id: str, candidates: tuple[MatchCandidate, ...]
    ) -> MatchResult:
        reasons, competitors = self._ambiguous[transaction_id]
        return MatchResult(
            transaction_id=transaction_id,
            status=MatchStatus.AMBIGUOUS,
            candidates=candidates,
            ambiguity_reasons=tuple(reasons),
            competing_candidates=tuple(competitors),
        )

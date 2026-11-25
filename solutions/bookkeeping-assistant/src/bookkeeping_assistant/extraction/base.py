"""The `Extractor` protocol and the offline `FakeExtractor`."""

import re
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import ValidationError

from bookkeeping_assistant.extraction.schema import (
    DocumentExtraction,
    ExtractionErrorCode,
    ExtractionResult,
    ExtractionStatus,
)
from bookkeeping_assistant.ingestion import IngestedDocument

_FIXTURE_NAME: Final = re.compile(r"[0-9a-f]{64}\.json")


class Extractor(Protocol):
    """Reads one document into a `DocumentExtraction`.

    Contract:
    - Per-document problems (unreadable file, refusal, timeout, invalid output) are returned as
      `ExtractionResult(status=FAILED)`, never raised. One bad document must not fail a review.
    - The output is untrusted. Callers interpret it only through deterministic normalization.
    """

    async def extract(self, document: IngestedDocument) -> ExtractionResult: ...


class FakeExtractor:
    """Offline extractor that returns pre-recorded JSON keyed by the document's SHA-256.

    Fixture JSON is validated exactly like real model output (`DocumentExtraction`, strict). A
    malformed fixture therefore behaves like a malformed model response: a failed result.
    """

    MODEL_NAME: Final = "fake"

    def __init__(self, fixtures: Mapping[str, str]) -> None:
        """`fixtures` maps lowercase hex SHA-256 digests to raw JSON text."""
        self._fixtures: Mapping[str, str] = MappingProxyType(dict(fixtures))

    @classmethod
    def from_directory(cls, directory: Path) -> "FakeExtractor":
        """Load ``<sha256>.json`` files. Any other file name is a configuration error."""
        fixtures: dict[str, str] = {}
        for path in sorted(directory.iterdir()):
            if not _FIXTURE_NAME.fullmatch(path.name):
                raise ValueError(f"unexpected file in extraction fixtures: {path.name}")
            fixtures[path.name.removesuffix(".json")] = path.read_text(encoding="utf-8")
        return cls(fixtures)

    async def extract(self, document: IngestedDocument) -> ExtractionResult:
        document_id = document.metadata.id
        raw = self._fixtures.get(document.metadata.sha256)
        if raw is None:
            return self._failed(document_id, ExtractionErrorCode.NO_FIXTURE)
        try:
            extraction = DocumentExtraction.model_validate_json(raw)
        except ValidationError:
            return self._failed(document_id, ExtractionErrorCode.INVALID_OUTPUT)
        return ExtractionResult(
            document_id=document_id,
            status=ExtractionStatus.OK,
            extraction=extraction,
            model=self.MODEL_NAME,
        )

    def _failed(self, document_id: str, code: ExtractionErrorCode) -> ExtractionResult:
        return ExtractionResult(
            document_id=document_id,
            status=ExtractionStatus.FAILED,
            error_code=code,
            model=self.MODEL_NAME,
        )

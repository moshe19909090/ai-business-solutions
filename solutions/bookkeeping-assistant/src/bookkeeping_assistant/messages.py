"""Message catalogs: reviewer-facing text rendered from stable codes.

Findings and evidence store codes and locale-neutral parameters. The words a reviewer reads
come from a JSON catalog per locale in `locales/`, so a new language is a new catalog file and
never a change to the report schema.

Key scheme (every key the code can ask for is listed by `required_keys`):

- ``finding.<FindingMessage>.title`` and ``finding.<FindingMessage>.explanation``
- ``hint.<FindingHint>``, ``next_action.<NextAction>``, ``severity.<Severity>``
- ``evidence.<EvidenceField>`` (evidence labels)
- ``issue.<NormalizationIssue>`` (reasons a document is incomplete)
- ``exclusion.<DocumentExclusion>`` (why a document was not used for matching)
- ``report.disclaimer``

Templates use ``{name}`` placeholders and nothing else (no format specs, no attribute access).
Parameters are locale-neutral tokens (see `models.MessageParams`). In a right-to-left catalog
every substituted value is wrapped in a left-to-right isolate (U+2066 ... U+2069), so amounts
such as ``-12.40`` and codes such as ``USD`` keep their order inside a Hebrew sentence.

A key missing from a locale falls back to English. The catalog tests require every locale to
be complete, so the fallback only protects against a crash, never replaces a translation.
"""

import json
import re
from collections.abc import Mapping
from enum import StrEnum
from functools import lru_cache
from importlib import resources
from typing import Final, Literal

from pydantic import Field

from bookkeeping_assistant.models import (
    DomainModel,
    EvidenceField,
    FindingHint,
    FindingMessage,
    FrozenStrMap,
    NextAction,
    NormalizationIssue,
    Severity,
)

TextDirection = Literal["ltr", "rtl"]

LEFT_TO_RIGHT_ISOLATE: Final = "⁦"
POP_DIRECTIONAL_ISOLATE: Final = "⁩"

_PLACEHOLDER: Final = re.compile(r"\{([a-z][a-z0-9_]*)\}")
_KEY: Final = re.compile(r"[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+")


class Locale(StrEnum):
    """Locales with a catalog. Report language only; never affects parsing or findings."""

    EN = "en"
    HE = "he"


DEFAULT_LOCALE: Final = Locale.EN

DOCUMENT_EXCLUSION_CODES: Final = (
    "unreadable",
    "total_unavailable",
    "total_not_positive",
    "date_unavailable",
    "currency_mismatch",
)
"""Values of `matching.DocumentExclusion`, listed here so this module does not import the
matcher. A test keeps the two in sync."""


class MissingMessageError(LookupError):
    """No catalog, including the English fallback, defines the requested key."""


class _CatalogFile(DomainModel):
    locale: Locale
    direction: TextDirection
    messages: dict[str, str] = Field(default_factory=dict)


class Catalog(DomainModel):
    """A validated, read-only catalog for one locale."""

    locale: Locale
    direction: TextDirection
    messages: FrozenStrMap


def placeholders(template: str) -> frozenset[str]:
    """The placeholder names a template uses."""
    return frozenset(_PLACEHOLDER.findall(template))


def _check_template(key: str, template: str) -> None:
    if not _KEY.fullmatch(key):
        raise ValueError(f"invalid catalog key: {key!r}")
    if not template.strip():
        raise ValueError(f"empty template for {key!r}")
    if "{" in _PLACEHOLDER.sub("", template) or "}" in _PLACEHOLDER.sub("", template):
        raise ValueError(f"template for {key!r} has a brace that is not a {{name}} placeholder")


def parse_catalog(text: str) -> Catalog:
    """Validate a catalog's JSON text. Raises `ValueError` (or `ValidationError`) if invalid."""
    data = _CatalogFile.model_validate(json.loads(text))
    for key, template in data.messages.items():
        _check_template(key, template)
    return Catalog(locale=data.locale, direction=data.direction, messages=data.messages)


@lru_cache(maxsize=len(Locale))
def load_catalog(locale: Locale) -> Catalog:
    """The packaged catalog for `locale`, read once and cached."""
    text = resources.files("bookkeeping_assistant").joinpath("locales", f"{locale.value}.json")
    catalog = parse_catalog(text.read_text(encoding="utf-8"))
    if catalog.locale is not locale:
        raise ValueError(f"catalog file {locale.value}.json declares locale {catalog.locale}")
    return catalog


def text_direction(locale: Locale | str) -> TextDirection:
    """Writing direction of the locale's catalog, for a UI or document renderer."""
    return load_catalog(Locale(locale)).direction


def render_template(
    template: str, params: Mapping[str, str], *, direction: TextDirection = "ltr"
) -> str:
    """Substitute ``{name}`` placeholders. Raises `ValueError` if a parameter is missing.

    Extra parameters are allowed, so a title can use fewer parameters than its explanation.
    """

    def substitute(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            raise ValueError(f"missing message parameter: {name}")
        value = params[name]
        if direction == "rtl":
            return f"{LEFT_TO_RIGHT_ISOLATE}{value}{POP_DIRECTIONAL_ISOLATE}"
        return value

    return _PLACEHOLDER.sub(substitute, template)


def render(
    key: str, params: Mapping[str, str] | None = None, *, locale: Locale | str = DEFAULT_LOCALE
) -> str:
    """Render `key` in `locale`, falling back to English when the locale lacks the key."""
    catalog = load_catalog(Locale(locale))
    template = catalog.messages.get(key)
    if template is None:
        catalog = load_catalog(DEFAULT_LOCALE)
        template = catalog.messages.get(key)
    if template is None:
        raise MissingMessageError(key)
    return render_template(template, params or {}, direction=catalog.direction)


def finding_title_key(message: FindingMessage) -> str:
    return f"finding.{message.value}.title"


def finding_explanation_key(message: FindingMessage) -> str:
    return f"finding.{message.value}.explanation"


def required_keys() -> frozenset[str]:
    """Every catalog key the code can request. Each locale must define all of them."""
    keys = {"report.disclaimer"}
    for message in FindingMessage:
        keys.add(finding_title_key(message))
        keys.add(finding_explanation_key(message))
    keys.update(f"hint.{hint.value}" for hint in FindingHint)
    keys.update(f"next_action.{action.value}" for action in NextAction)
    keys.update(f"severity.{severity.value}" for severity in Severity)
    keys.update(f"evidence.{field.value}" for field in EvidenceField)
    keys.update(f"issue.{issue.value}" for issue in NormalizationIssue)
    keys.update(f"exclusion.{code}" for code in DOCUMENT_EXCLUSION_CODES)
    return frozenset(keys)

"""Rendering mechanics of `messages.py`. Catalog wording and completeness are in test_messages."""

import json

import pytest
from pydantic import ValidationError

from bookkeeping_assistant.matching import DocumentExclusion
from bookkeeping_assistant.messages import (
    DOCUMENT_EXCLUSION_CODES,
    LEFT_TO_RIGHT_ISOLATE,
    POP_DIRECTIONAL_ISOLATE,
    Locale,
    MissingMessageError,
    finding_explanation_key,
    finding_title_key,
    load_catalog,
    parse_catalog,
    placeholders,
    render,
    render_template,
    required_keys,
    text_direction,
)
from bookkeeping_assistant.models import FINDING_MESSAGE_PARAMS, FindingMessage

MISSING_PARAMS = {"amount": "-120.00", "currency": "ILS", "date": "2026-03-05", "threshold": "250"}


def _catalog_text(messages: dict[str, str], *, locale: str = "en", direction: str = "ltr") -> str:
    return json.dumps({"locale": locale, "direction": direction, "messages": messages})


def test_render_substitutes_params_in_english() -> None:
    text = render(finding_explanation_key(FindingMessage.MISSING_DOCUMENT), MISSING_PARAMS)

    assert "-120.00 ILS" in text
    assert "2026-03-05" in text
    assert LEFT_TO_RIGHT_ISOLATE not in text


def test_rtl_catalog_isolates_every_substituted_value() -> None:
    text = render_template("סכום {amount} {currency}", MISSING_PARAMS, direction="rtl")

    assert text == (
        f"סכום {LEFT_TO_RIGHT_ISOLATE}-120.00{POP_DIRECTIONAL_ISOLATE}"
        f" {LEFT_TO_RIGHT_ISOLATE}ILS{POP_DIRECTIONAL_ISOLATE}"
    )


def test_missing_key_in_locale_falls_back_to_english() -> None:
    key = finding_title_key(FindingMessage.UNMATCHED_DOCUMENT)

    assert render(key, locale=Locale.HE) == render(key, locale=Locale.EN)


def test_unknown_key_raises() -> None:
    with pytest.raises(MissingMessageError):
        render("finding.not_a_message.title")


def test_missing_param_raises() -> None:
    with pytest.raises(ValueError, match="missing message parameter: threshold"):
        render(
            finding_explanation_key(FindingMessage.MISSING_DOCUMENT),
            {k: v for k, v in MISSING_PARAMS.items() if k != "threshold"},
        )


def test_extra_params_are_allowed_for_shorter_templates() -> None:
    title = render(finding_title_key(FindingMessage.MISSING_DOCUMENT), MISSING_PARAMS)

    assert title


def test_unknown_locale_is_rejected() -> None:
    with pytest.raises(ValueError, match="'fr' is not a valid Locale"):
        render("report.disclaimer", locale="fr")


def test_text_direction() -> None:
    assert text_direction("en") == "ltr"
    assert text_direction(Locale.HE) == "rtl"


def test_packaged_catalogs_load_and_declare_their_locale() -> None:
    for locale in Locale:
        assert load_catalog(locale).locale is locale


def test_english_starter_catalog_defines_every_required_key() -> None:
    assert required_keys() <= set(load_catalog(Locale.EN).messages)


def test_english_finding_templates_use_only_contract_params() -> None:
    messages = load_catalog(Locale.EN).messages
    for message, allowed in FINDING_MESSAGE_PARAMS.items():
        for key in (finding_title_key(message), finding_explanation_key(message)):
            assert placeholders(messages[key]) <= allowed, key


def test_exclusion_codes_match_the_matcher() -> None:
    assert tuple(e.value for e in DocumentExclusion) == DOCUMENT_EXCLUSION_CODES


def test_placeholders() -> None:
    assert placeholders("{a} and {b_2} and {a}") == frozenset({"a", "b_2"})


@pytest.mark.parametrize(
    ("messages", "error"),
    [
        pytest.param({"Finding.x": "x"}, "invalid catalog key", id="bad-key"),
        pytest.param({"report": "x"}, "invalid catalog key", id="key-without-dot"),
        pytest.param({"report.x": "  "}, "empty template", id="empty"),
        pytest.param({"report.x": "{amount:.2f}"}, "brace", id="format-spec"),
        pytest.param({"report.x": "{obj.attr}"}, "brace", id="attribute-access"),
        pytest.param({"report.x": "{0}"}, "brace", id="positional"),
        pytest.param({"report.x": "a } b"}, "brace", id="stray-brace"),
    ],
)
def test_parse_catalog_rejects_unsafe_templates(messages: dict[str, str], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        parse_catalog(_catalog_text(messages))


@pytest.mark.parametrize(
    "data",
    [
        pytest.param({"locale": "fr", "direction": "ltr", "messages": {}}, id="locale"),
        pytest.param({"locale": "he", "direction": "auto", "messages": {}}, id="direction"),
        pytest.param({"locale": "en", "direction": "ltr", "extra": 1}, id="unknown-field"),
    ],
)
def test_parse_catalog_rejects_invalid_structure(data: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        parse_catalog(json.dumps(data))


def test_catalog_messages_are_read_only() -> None:
    messages = load_catalog(Locale.EN).messages

    with pytest.raises(TypeError):
        messages["report.disclaimer"] = "changed"  # type: ignore[index]

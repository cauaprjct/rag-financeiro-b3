"""Property 22: Credenciais nunca aparecem em mensagens."""

import os
from unittest.mock import patch

from hypothesis import given, settings, assume
from hypothesis.strategies import text, characters

from core.config import load_config, sanitize_message

_safe_text = text(
    alphabet=characters(blacklist_characters="\x00*", blacklist_categories=("Cs",)),
    min_size=1,
    max_size=80,
)


@settings(max_examples=100)
@given(api_key=_safe_text, message_body=_safe_text)
def test_sanitize_removes_credential(api_key, message_body):
    assume(api_key.strip())
    msg = f"Error occurred: {api_key} details"
    env = {"GROQ_API_KEY": api_key, "GEMINI_API_KEY": ""}
    with patch.dict(os.environ, env, clear=True):
        cfg = load_config()
    sanitized = sanitize_message(msg, cfg)
    assert api_key not in sanitized


@settings(max_examples=100)
@given(groq_key=_safe_text, gemini_key=_safe_text)
def test_both_credentials_removed(groq_key, gemini_key):
    assume(groq_key.strip())
    assume(gemini_key.strip())
    msg = f"keys: {groq_key} and {gemini_key}"
    env = {"GROQ_API_KEY": groq_key, "GEMINI_API_KEY": gemini_key}
    with patch.dict(os.environ, env, clear=True):
        cfg = load_config()
    sanitized = sanitize_message(msg, cfg)
    assert groq_key not in sanitized
    assert gemini_key not in sanitized

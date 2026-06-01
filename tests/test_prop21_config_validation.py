"""Property 21: Validação de configuração aceita exatamente entradas válidas."""

import os
from unittest.mock import patch

import pytest
from hypothesis import given, settings, assume
from hypothesis.strategies import integers, text, characters

from core.config import load_config, ConfigError


@settings(max_examples=100)
@given(
    chunk_size=integers(min_value=1, max_value=10000),
    overlap=integers(min_value=0, max_value=9999),
    top_k=integers(min_value=1, max_value=100),
    emb_model=text(
        alphabet=characters(blacklist_characters="\x00", blacklist_categories=("Cs",)),
        min_size=1,
        max_size=50,
    ),
)
def test_valid_config_accepted(chunk_size, overlap, top_k, emb_model):
    assume(overlap < chunk_size)
    assume(emb_model.strip())
    env = {
        "GROQ_API_KEY": "key123",
        "CHUNK_SIZE": str(chunk_size),
        "CHUNK_OVERLAP": str(overlap),
        "TOP_K": str(top_k),
        "EMBEDDING_MODEL_ID": emb_model,
    }
    with patch.dict(os.environ, env, clear=True):
        cfg = load_config()
    assert cfg.chunk_size == chunk_size
    assert cfg.chunk_overlap == overlap
    assert cfg.top_k == top_k


@settings(max_examples=100)
@given(chunk_size=integers(max_value=0))
def test_invalid_chunk_size_rejected(chunk_size):
    env = {"GROQ_API_KEY": "k", "CHUNK_SIZE": str(chunk_size)}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError, match="CHUNK_SIZE"):
            load_config()


@settings(max_examples=100)
@given(top_k=integers(max_value=0))
def test_invalid_top_k_rejected(top_k):
    env = {"GROQ_API_KEY": "k", "TOP_K": str(top_k)}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError, match="TOP_K"):
            load_config()


@settings(max_examples=100)
@given(
    chunk_size=integers(min_value=1, max_value=1000),
    overlap=integers(min_value=0, max_value=10000),
)
def test_overlap_gte_chunk_size_rejected(chunk_size, overlap):
    assume(overlap >= chunk_size)
    env = {"GROQ_API_KEY": "k", "CHUNK_SIZE": str(chunk_size), "CHUNK_OVERLAP": str(overlap)}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
            load_config()


@settings(max_examples=100)
@given(overlap=integers(max_value=-1))
def test_negative_overlap_rejected(overlap):
    env = {"GROQ_API_KEY": "k", "CHUNK_OVERLAP": str(overlap)}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(ConfigError, match="CHUNK_OVERLAP"):
            load_config()

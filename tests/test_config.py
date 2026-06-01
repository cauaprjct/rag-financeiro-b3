"""Testes de exemplo do Config_Module."""

import os
from unittest.mock import patch

import pytest

from core.config import load_config, ConfigError


class TestPrecedence:
    def test_env_overrides_secrets(self):
        secrets = {"GROQ_API_KEY": "from_secrets", "CHUNK_SIZE": "100"}
        env = {"GROQ_API_KEY": "from_env", "CHUNK_SIZE": "200"}
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config(secrets)
        assert cfg.groq_api_key == "from_env"
        assert cfg.chunk_size == 200

    def test_secrets_used_when_env_absent(self):
        secrets = {"GROQ_API_KEY": "from_secrets", "TOP_K": "10"}
        with patch.dict(os.environ, {}, clear=True):
            cfg = load_config(secrets)
        assert cfg.groq_api_key == "from_secrets"
        assert cfg.top_k == 10


class TestCredentialRequired:
    def test_no_credential_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="Credencial ausente"):
                load_config()

    def test_only_gemini_key_works(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "gk"}, clear=True):
            cfg = load_config()
        assert cfg.gemini_api_key == "gk"


class TestInvalidParam:
    def test_invalid_chunk_size_message(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "k", "CHUNK_SIZE": "abc"}, clear=True):
            with pytest.raises(ConfigError, match="CHUNK_SIZE"):
                load_config()


class TestRerankConfig:
    def test_rerank_disabled_by_default(self):
        with patch.dict(os.environ, {"GROQ_API_KEY": "k"}, clear=True):
            cfg = load_config()
        assert cfg.rerank_enabled is False

    def test_rerank_enabled_empty_model_raises(self):
        env = {"GROQ_API_KEY": "k", "RERANK_ENABLED": "true", "RERANKER_MODEL_ID": "  "}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="RERANKER_MODEL_ID"):
                load_config()

    def test_rerank_candidates_lt_top_k_raises(self):
        env = {
            "GROQ_API_KEY": "k",
            "RERANK_ENABLED": "true",
            "TOP_K": "10",
            "RERANK_CANDIDATES_N": "5",
        }
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(ConfigError, match="RERANK_CANDIDATES_N"):
                load_config()

    def test_rerank_valid_config(self):
        env = {
            "GROQ_API_KEY": "k",
            "RERANK_ENABLED": "true",
            "TOP_K": "5",
            "RERANK_CANDIDATES_N": "20",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
        assert cfg.rerank_enabled is True
        assert cfg.rerank_candidates_n == 20

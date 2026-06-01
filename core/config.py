from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(Exception):
    pass


_CREDENTIAL_KEYS = ("GROQ_API_KEY", "GEMINI_API_KEY")


@dataclass(frozen=True)
class AppConfig:
    groq_api_key: str
    gemini_api_key: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    embedding_model_id: str
    rerank_enabled: bool
    reranker_model_id: str
    rerank_candidates_n: int
    device: str


def _read(key: str, secrets: dict | None) -> str | None:
    val = os.environ.get(key)
    if val is not None:
        return val
    if secrets and key in secrets:
        return str(secrets[key])
    return None


def load_config(secrets: dict | None = None) -> AppConfig:
    """Load config with precedence: env > secrets. Raises ConfigError on invalid input."""
    groq_key = _read("GROQ_API_KEY", secrets) or ""
    gemini_key = _read("GEMINI_API_KEY", secrets) or ""
    if not groq_key and not gemini_key:
        raise ConfigError("Credencial ausente: GROQ_API_KEY ou GEMINI_API_KEY deve ser fornecida")

    chunk_size = _parse_int("CHUNK_SIZE", secrets, default=512)
    chunk_overlap = _parse_int("CHUNK_OVERLAP", secrets, default=64)
    top_k = _parse_int("TOP_K", secrets, default=5)
    embedding_model_id = _read("EMBEDDING_MODEL_ID", secrets) or "intfloat/multilingual-e5-small"

    rerank_enabled = _parse_bool("RERANK_ENABLED", secrets, default=False)
    reranker_model_id = _read("RERANKER_MODEL_ID", secrets) or "BAAI/bge-reranker-v2-m3"
    rerank_candidates_n = _parse_int("RERANK_CANDIDATES_N", secrets, default=20)
    device = _read("DEVICE", secrets) or "auto"

    _validate(
        chunk_size,
        chunk_overlap,
        top_k,
        embedding_model_id,
        rerank_enabled,
        reranker_model_id,
        rerank_candidates_n,
        device,
    )

    return AppConfig(
        groq_api_key=groq_key,
        gemini_api_key=gemini_key,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        top_k=top_k,
        embedding_model_id=embedding_model_id,
        rerank_enabled=rerank_enabled,
        reranker_model_id=reranker_model_id,
        rerank_candidates_n=rerank_candidates_n,
        device=device,
    )


def _parse_int(key: str, secrets: dict | None, *, default: int) -> int:
    raw = _read(key, secrets)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ConfigError(f"Parâmetro inválido: {key} deve ser inteiro, recebeu '{raw}'")


def _parse_bool(key: str, secrets: dict | None, *, default: bool) -> bool:
    raw = _read(key, secrets)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def _validate(
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    embedding_model_id: str,
    rerank_enabled: bool,
    reranker_model_id: str,
    rerank_candidates_n: int,
    device: str,
) -> None:
    if chunk_size < 1:
        raise ConfigError("Parâmetro inválido: CHUNK_SIZE deve ser >= 1")
    if top_k < 1:
        raise ConfigError("Parâmetro inválido: TOP_K deve ser >= 1")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ConfigError("Parâmetro inválido: CHUNK_OVERLAP deve ser >= 0 e < CHUNK_SIZE")
    if not embedding_model_id.strip():
        raise ConfigError("Parâmetro inválido: EMBEDDING_MODEL_ID não pode ser vazio")
    if device not in ("auto", "cpu", "cuda"):
        raise ConfigError("Parâmetro inválido: DEVICE deve ser 'auto', 'cpu' ou 'cuda'")
    if rerank_enabled:
        if not reranker_model_id.strip():
            raise ConfigError(
                "Parâmetro inválido: RERANKER_MODEL_ID não pode ser vazio "
                "quando reranking habilitado"
            )
        if rerank_candidates_n < top_k:
            raise ConfigError("Parâmetro inválido: RERANK_CANDIDATES_N deve ser >= TOP_K")


def sanitize_message(message: str, cfg: AppConfig) -> str:
    """Remove credential values from a message string."""
    result = message
    for val in (cfg.groq_api_key, cfg.gemini_api_key):
        if val:
            result = result.replace(val, "***")
    return result

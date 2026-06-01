from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    index: int
    char_count: int


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    document_name: str
    page_number: int
    snippet: str


@dataclass
class RetrievalResult:
    scored_chunks: list[ScoredChunk] = field(default_factory=list)
    message: str = ""


@dataclass(frozen=True)
class DocumentInfo:
    name: str
    chunk_count: int


@dataclass
class DeleteResult:
    document_name: str
    deleted_count: int
    message: str = ""


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass
class IngestionResult:
    document_name: str
    pages: list[PageText] = field(default_factory=list)
    success: bool = True
    error: str = ""


@dataclass
class EmbedBatchResult:
    embeddings: list[Optional[list[float]]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvalExample:
    question: str
    reference_answer: str
    expected_context: str = ""


@dataclass
class ExampleMetrics:
    context_precision: float = 0.0
    faithfulness: float = 0.0
    response_relevancy: float = 0.0
    evaluated: bool = True
    error: str = ""


@dataclass
class EvaluationReport:
    total: int = 0
    evaluated: int = 0
    context_precision: float = 0.0
    faithfulness: float = 0.0
    response_relevancy: float = 0.0
    embedding_model_id: str = ""
    llm_model_id: str = ""
    examples: list[ExampleMetrics] = field(default_factory=list)
    message: str = ""

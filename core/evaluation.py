"""Evaluation module: LLM-as-judge metrics, response relevancy, dataset loader."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

from core.models import EvalExample, ExampleMetrics, EvaluationReport
from prompts.judge import CONTEXT_PRECISION_PROMPT, FAITHFULNESS_PROMPT

_EVAL_TIMEOUT_S = 60.0


class EvalDatasetError(Exception):
    pass


def load_eval_dataset(path: str | Path) -> list[EvalExample]:
    """Load eval dataset from local JSON. Validates required fields."""
    p = Path(path)
    if not p.exists():
        raise EvalDatasetError(f"Arquivo de dataset não encontrado: {p}")

    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvalDatasetError(f"JSON malformado em {p}: {e}")

    if not isinstance(raw, list):
        raise EvalDatasetError(f"Dataset deve ser uma lista de exemplos: {p}")

    examples: list[EvalExample] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise EvalDatasetError(f"Exemplo {i} não é um objeto: {p}")
        question = item.get("question")
        reference = item.get("reference_answer")
        if not question or not str(question).strip():
            raise EvalDatasetError(f"Exemplo {i}: campo obrigatório 'question' ausente ou vazio")
        if not reference or not str(reference).strip():
            raise EvalDatasetError(
                f"Exemplo {i}: campo obrigatório 'reference_answer' ausente ou vazio"
            )
        examples.append(
            EvalExample(
                question=str(question),
                reference_answer=str(reference),
                expected_context=str(item.get("expected_context", "")),
            )
        )
    return examples


def _parse_score(text: str) -> float:
    """Parse a 0-1 float from LLM response. Clamps to [0, 1]."""
    for token in text.replace("\n", " ").split():
        try:
            val = float(token.strip().rstrip(".,"))
            return max(0.0, min(1.0, val))
        except ValueError:
            continue
    return 0.0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (na * nb)))


class EvaluationModule:
    """Computes Context Precision, Faithfulness (LLM judge) and Response Relevancy."""

    def __init__(self, llm_judge, embedding_module, eval_dir: str = "data/eval") -> None:
        self._llm = llm_judge
        self._emb = embedding_module
        self._eval_dir = Path(eval_dir)

    def evaluate(
        self,
        dataset: list[EvalExample],
        answer_fn: Callable[[str], tuple[str, str]],
    ) -> EvaluationReport:
        """Evaluate dataset. answer_fn(question) -> (answer, retrieved_context)."""
        report = EvaluationReport(
            total=len(dataset),
            embedding_model_id=self._emb.model_id,
            llm_model_id=self._llm.model_id,
        )

        if not dataset:
            report.message = "Dataset vazio: nenhuma métrica calculada."
            self._persist(report)
            return report

        for ex in dataset:
            metrics = self._evaluate_one(ex, answer_fn)
            report.examples.append(metrics)

        evaluated = [m for m in report.examples if m.evaluated]
        report.evaluated = len(evaluated)

        if not evaluated:
            report.message = "Nenhum exemplo pôde ser avaliado."
            self._persist(report)
            return report

        report.context_precision = sum(m.context_precision for m in evaluated) / len(evaluated)
        report.faithfulness = sum(m.faithfulness for m in evaluated) / len(evaluated)
        report.response_relevancy = sum(m.response_relevancy for m in evaluated) / len(evaluated)
        self._persist(report)
        return report

    def _evaluate_one(self, ex: EvalExample, answer_fn) -> ExampleMetrics:
        try:
            answer, context = answer_fn(ex.question)

            cp_raw = self._llm.complete(
                CONTEXT_PRECISION_PROMPT.format(question=ex.question, context=context),
                timeout_s=_EVAL_TIMEOUT_S,
            )
            faith_raw = self._llm.complete(
                FAITHFULNESS_PROMPT.format(context=context, answer=answer),
                timeout_s=_EVAL_TIMEOUT_S,
            )

            ans_vec = self._emb.embed_query(answer)
            ref_vec = self._emb.embed_query(ex.reference_answer)
            relevancy = _cosine(ans_vec, ref_vec) if ans_vec and ref_vec else 0.0

            return ExampleMetrics(
                context_precision=_parse_score(cp_raw),
                faithfulness=_parse_score(faith_raw),
                response_relevancy=relevancy,
                evaluated=True,
            )
        except Exception as e:
            return ExampleMetrics(evaluated=False, error=str(e))

    def _persist(self, report: EvaluationReport) -> None:
        self._eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self._eval_dir / f"report_{timestamp}.json"
        path.write_text(
            json.dumps(_report_to_dict(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _report_to_dict(report: EvaluationReport) -> dict:
    return {
        "total": report.total,
        "evaluated": report.evaluated,
        "context_precision": report.context_precision,
        "faithfulness": report.faithfulness,
        "response_relevancy": report.response_relevancy,
        "embedding_model_id": report.embedding_model_id,
        "llm_model_id": report.llm_model_id,
        "message": report.message,
        "examples": [
            {
                "context_precision": m.context_precision,
                "faithfulness": m.faithfulness,
                "response_relevancy": m.response_relevancy,
                "evaluated": m.evaluated,
                "error": m.error,
            }
            for m in report.examples
        ],
    }

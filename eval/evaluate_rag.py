"""
RAG evaluation harness.
Runs the pipeline on a dataset and computes heuristic metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import numpy as np
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vector_store import VectorStore  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None


DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be a list of objects")
        return data
    raise ValueError("Dataset must be .jsonl or .json")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    return _normalize_text(text).split()


def _f1_score(pred: str, gold: str) -> float:
    pred_tokens = _tokenize(pred)
    gold_tokens = _tokenize(gold)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1
    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(gold_tokens), 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_text(pred) == _normalize_text(gold) else 0.0


def _split_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class EvalConfig:
    dataset_path: Path
    output_root: Path
    top_k: int
    use_local_llm: bool
    local_llm_base_url: str
    local_llm_model: str
    use_openai_embeddings: bool
    embedding_model: str
    compare_baseline: bool


class Embedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, normalize_embeddings=True))


def _build_llm(config: EvalConfig):
    if ChatOpenAI is None:
        raise ImportError("langchain-openai is required to run LLM baselines")
    if config.use_local_llm:
        return ChatOpenAI(
            model=config.local_llm_model,
            temperature=0.0,
            base_url=config.local_llm_base_url,
            api_key=os.getenv("LOCAL_LLM_API_KEY", "lm-studio"),
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when not using a local LLM")
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=api_key)


def _invoke_llm(llm, question: str) -> str:
    prompt = f"Answer the question concisely.\n\nQuestion: {question}\nAnswer:"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))


def _compute_groundedness(
    answer: str,
    contexts: List[str],
    embedder: Embedder,
) -> Optional[float]:
    if not answer or not contexts:
        return None
    sentences = _split_sentences(answer)
    if not sentences:
        return None
    sentence_embeds = embedder.embed(sentences)
    context_embeds = embedder.embed(contexts)
    scores = []
    for sent_vec in sentence_embeds:
        sims = np.dot(context_embeds, sent_vec)
        scores.append(float(np.max(sims)))
    return float(np.mean(scores)) if scores else None


def _compute_similarity(a: str, b: str, embedder: Embedder) -> Optional[float]:
    if not a or not b:
        return None
    vecs = embedder.embed([a, b])
    return _cosine_similarity(vecs[0], vecs[1])


def _summarize_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def avg(key: str) -> Optional[float]:
        values = [row[key] for row in rows if row.get(key) is not None]
        if not values:
            return None
        return float(np.mean(values))

    return {
        "count": len(rows),
        "exact_match": avg("exact_match"),
        "f1": avg("f1"),
        "semantic_similarity": avg("semantic_similarity"),
        "relevance": avg("relevance"),
        "groundedness": avg("groundedness"),
        "hallucination_rate": avg("hallucination_rate"),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_eval(config: EvalConfig) -> Dict[str, Any]:
    dataset = _load_dataset(config.dataset_path)
    if not dataset:
        raise ValueError("Dataset is empty")

    vector_store = VectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "ancient-egypt-rag"),
        use_openai=config.use_openai_embeddings,
    )
    vectorstore = vector_store.load_vector_store()

    rag = RAGPipeline(
        vectorstore,
        top_k=config.top_k,
        use_local_llm=config.use_local_llm,
        local_llm_base_url=config.local_llm_base_url,
        model_name=config.local_llm_model if config.use_local_llm else None,
    )

    baseline_llm = _build_llm(config) if config.compare_baseline else None
    embedder = Embedder(config.embedding_model)

    rag_rows: List[Dict[str, Any]] = []
    baseline_rows: List[Dict[str, Any]] = []

    for item in dataset:
        question = item.get("question", "").strip()
        if not question:
            continue
        ref_answer = item.get("reference_answer", "") or ""

        rag_result = rag.query_with_sources(question)
        rag_answer = rag_result.get("answer", "")

        docs = vectorstore.similarity_search(question, k=config.top_k)
        contexts = [doc.page_content for doc in docs]

        groundedness = _compute_groundedness(rag_answer, contexts, embedder)
        rag_row = {
            "id": item.get("id"),
            "question": question,
            "reference_answer": ref_answer,
            "answer": rag_answer,
            "exact_match": _exact_match(rag_answer, ref_answer) if ref_answer else None,
            "f1": _f1_score(rag_answer, ref_answer) if ref_answer else None,
            "semantic_similarity": _compute_similarity(rag_answer, ref_answer, embedder),
            "relevance": _compute_similarity(rag_answer, question, embedder),
            "groundedness": groundedness,
            "hallucination_rate": None if groundedness is None else float(max(0.0, 1.0 - groundedness)),
            "context_snippets": [c[:300] + "..." for c in contexts],
        }
        rag_rows.append(rag_row)

        if baseline_llm:
            baseline_answer = _invoke_llm(baseline_llm, question)
            baseline_row = {
                "id": item.get("id"),
                "question": question,
                "reference_answer": ref_answer,
                "answer": baseline_answer,
                "exact_match": _exact_match(baseline_answer, ref_answer) if ref_answer else None,
                "f1": _f1_score(baseline_answer, ref_answer) if ref_answer else None,
                "semantic_similarity": _compute_similarity(baseline_answer, ref_answer, embedder),
                "relevance": _compute_similarity(baseline_answer, question, embedder),
                "groundedness": None,
                "hallucination_rate": None,
            }
            baseline_rows.append(baseline_row)

    summary = {
        "rag": _summarize_metrics(rag_rows),
        "baseline": _summarize_metrics(baseline_rows) if baseline_rows else None,
    }

    return {
        "rag_rows": rag_rows,
        "baseline_rows": baseline_rows,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--dataset", required=True, help="Path to JSONL/JSON dataset")
    parser.add_argument("--top-k", type=int, default=4, help="Number of documents to retrieve")
    parser.add_argument("--use-local-llm", action="store_true", help="Use local LM Studio")
    parser.add_argument(
        "--local-llm-base-url",
        default=os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1"),
    )
    parser.add_argument(
        "--local-llm-model",
        default=os.getenv("LOCAL_LLM_MODEL", "local-model"),
    )
    parser.add_argument(
        "--use-openai-embeddings",
        action="store_true",
        help="Use OpenAI embeddings (must match how your index was built)",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformers model for evaluation embeddings",
    )
    parser.add_argument(
        "--compare-baseline",
        action="store_true",
        help="Also run a non-RAG baseline with the same LLM",
    )

    args = parser.parse_args()

    config = EvalConfig(
        dataset_path=Path(args.dataset),
        output_root=Path("eval/results"),
        top_k=args.top_k,
        use_local_llm=args.use_local_llm,
        local_llm_base_url=args.local_llm_base_url,
        local_llm_model=args.local_llm_model,
        use_openai_embeddings=args.use_openai_embeddings,
        embedding_model=args.embedding_model,
        compare_baseline=args.compare_baseline,
    )

    run_output = run_eval(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = config.output_root / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.json"
    config_path.write_text(
        json.dumps({
            "dataset": str(config.dataset_path),
            "top_k": config.top_k,
            "use_local_llm": config.use_local_llm,
            "local_llm_base_url": config.local_llm_base_url,
            "local_llm_model": config.local_llm_model,
            "use_openai_embeddings": config.use_openai_embeddings,
            "embedding_model": config.embedding_model,
            "compare_baseline": config.compare_baseline,
        }, indent=2),
        encoding="utf-8",
    )

    (output_dir / "results.json").write_text(
        json.dumps(run_output["rag_rows"], indent=2), encoding="utf-8"
    )
    _write_csv(output_dir / "results.csv", run_output["rag_rows"])

    if run_output["baseline_rows"]:
        (output_dir / "baseline_results.json").write_text(
            json.dumps(run_output["baseline_rows"], indent=2), encoding="utf-8"
        )
        _write_csv(output_dir / "baseline_results.csv", run_output["baseline_rows"])

    (output_dir / "summary.json").write_text(
        json.dumps(run_output["summary"], indent=2), encoding="utf-8"
    )

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()

# RAG Evaluation

This folder contains a lightweight evaluation harness for the RAG pipeline.

## Dataset format
Provide a JSONL file where each line has at minimum:

```
{"id": "q1", "question": "...", "reference_answer": "..."}
```

Optional fields are allowed and will be carried through the results. Put your datasets in `eval/datasets/`.

## What gets evaluated
The script runs your RAG pipeline on each question and computes:

- Exact match and token-level F1 versus the reference answer.
- Semantic similarity between the answer and reference.
- Relevance score (answer vs. question similarity).
- Groundedness score (how well answer sentences align with retrieved context).
- Hallucination rate = `1 - groundedness`.

These are heuristic metrics and should be complemented with human review.

## Running the evaluation

```
python eval/evaluate_rag.py \
  --dataset eval/datasets/sample_eval.jsonl \
  --top-k 4 \
  --use-local-llm
```

If you want to use OpenAI instead of a local model, remove `--use-local-llm` and ensure `OPENAI_API_KEY` is set.

## Output
Each run writes a timestamped folder in `eval/results/` with:

- `results.json`: per-question outputs
- `results.csv`: per-question outputs in CSV form
- `summary.json`: aggregated metrics
- `config.json`: run configuration

## Plotting a summary chart

```
python eval/plot_results.py --run-dir eval/results/run_YYYYMMDD_HHMMSS
```

If you omit `--run-dir`, the script uses the latest run and writes `summary.png`
into that folder.

If you see `ModuleNotFoundError: No module named 'matplotlib'`, install it:

```
pip install matplotlib
```

## Notes
- You must have a Pinecone index available (set `PINECONE_API_KEY`, and `PINECONE_INDEX_NAME` if not the default).
- If you used local embeddings for indexing, pass `--use-openai-embeddings false` to keep it consistent.

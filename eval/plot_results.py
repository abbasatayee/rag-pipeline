"""Generate a comparison chart for RAG vs baseline results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt


def _load_summary(run_dir: Path) -> Dict[str, Dict[str, Optional[float]]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_run(results_dir: Path) -> Path:
    runs = sorted([p for p in results_dir.glob("run_*") if p.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No run_* directories found in {results_dir}")
    return runs[-1]


def _metric_or_none(summary: Dict[str, Optional[float]], *keys: str) -> Optional[float]:
    for key in keys:
        value = summary.get(key)
        if value is not None:
            return value
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot RAG evaluation summary")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to eval/results/run_*/ (defaults to latest)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path (defaults to <run-dir>/summary.png)",
    )

    args = parser.parse_args()

    results_root = Path("eval/results")
    run_dir = Path(args.run_dir) if args.run_dir else _latest_run(results_root)
    summary = _load_summary(run_dir)

    rag = summary.get("rag", {})
    baseline = summary.get("baseline") or {}

    metrics = ["Factual Accuracy", "Hallucination Rate", "Relevance Score"]

    rag_values = [
        _metric_or_none(rag, "f1", "exact_match") or 0.0,
        _metric_or_none(rag, "hallucination_rate") or 0.0,
        _metric_or_none(rag, "relevance") or 0.0,
    ]

    baseline_values = [
        _metric_or_none(baseline, "f1", "exact_match") or 0.0,
        _metric_or_none(baseline, "hallucination_rate") or 0.0,
        _metric_or_none(baseline, "relevance") or 0.0,
    ]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], baseline_values, width, label="Without RAG", color="#3d1f1f")
    ax.bar([i + width / 2 for i in x], rag_values, width, label="With RAG", color="#8b4a4a")

    ax.set_title("Results and Evaluation")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    output_path = Path(args.output) if args.output else run_dir / "summary.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved chart to {output_path}")


if __name__ == "__main__":
    main()

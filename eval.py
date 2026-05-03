"""
Evaluation pipeline for the Policy RAG system.

Computes Precision@K, Recall@K and NDCG@K against the 100 labeled examples.
Can compare multiple retrieval strategies side-by-side.

Usage:
    python eval.py                   # evaluate all strategies, K=5
    python eval.py --k 3             # use K=3
    python eval.py --split val       # only last-20 validation examples
    python eval.py --strategy hybrid # only one strategy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np

LABELED_DATA = Path("raw_policy/labeled_data.json")


# ── Metric helpers ────────────────────────────────────────────────────────────

def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return len(set(retrieved_ids[:k]) & relevant_ids) / k


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def dcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    score = 0.0
    for rank, pid in enumerate(retrieved_ids[:k], start=1):
        if pid in relevant_ids:
            score += 1.0 / np.log2(rank + 1)
    return score


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    ideal = sorted([1] * len(relevant_ids) + [0] * max(0, k - len(relevant_ids)), reverse=True)
    ideal_dcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal[:k]))
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, relevant_ids, k) / ideal_dcg


def f1_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    p = precision_at_k(retrieved_ids, relevant_ids, k)
    r = recall_at_k(retrieved_ids, relevant_ids, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# ── Strategy definitions ──────────────────────────────────────────────────────

def _get_policy_ids(chunks) -> list[str]:
    """Extract ordered policy IDs from retrieved chunks, deduplicating by ID."""
    seen: set[str] = set()
    ids: list[str] = []
    for c in chunks:
        pid = c.metadata.get("policy_id", "")
        if pid and pid not in seen:
            seen.add(pid)
            ids.append(pid)
    return ids


def strategy_dense_only(query: str, k: int):
    from rag_models import dense_retrieve
    chunks = dense_retrieve(query, n_results=k * 3)
    return _get_policy_ids(chunks)[:k]


def strategy_bm25_only(query: str, k: int):
    from rag_models import bm25_retrieve
    chunks = bm25_retrieve(query, n_results=k * 3)
    return _get_policy_ids(chunks)[:k]


def strategy_hybrid_rrf(query: str, k: int):
    from rag_models import retrieve_chunks
    chunks = retrieve_chunks(query, n_results=k * 3)
    return _get_policy_ids(chunks)[:k]


def strategy_hybrid_reranker(query: str, k: int):
    from rag_models import retrieve_chunks, rerank_chunks
    candidates = retrieve_chunks(query, n_results=20)
    chunks = rerank_chunks(query, candidates, top_n=k)
    return _get_policy_ids(chunks)[:k]


def strategy_full_pipeline(query: str, k: int):
    from rag_models import retrieve_chunks, rerank_chunks, lambdamart_rerank
    candidates = retrieve_chunks(query, n_results=20)
    chunks = rerank_chunks(query, candidates, top_n=k + 5)
    chunks = lambdamart_rerank(query, chunks, top_n=k)
    return _get_policy_ids(chunks)[:k]


STRATEGIES: dict[str, Callable] = {
    "dense_only": strategy_dense_only,
    "bm25_only": strategy_bm25_only,
    "hybrid_rrf": strategy_hybrid_rrf,
    "hybrid_reranker": strategy_hybrid_reranker,
    "full_pipeline": strategy_full_pipeline,
}


# ── Evaluation runner ─────────────────────────────────────────────────────────

def evaluate(
    examples: list[dict],
    strategy_fn: Callable,
    k: int,
    strategy_name: str,
) -> dict:
    p_scores, r_scores, ndcg_scores, f1_scores = [], [], [], []

    for ex in examples:
        query = ex["query"]
        relevant = set(ex["relevant_policy_ids"])
        try:
            retrieved = strategy_fn(query, k)
        except Exception as e:
            print(f"  [WARN] {ex['id']} failed: {e}")
            retrieved = []

        p_scores.append(precision_at_k(retrieved, relevant, k))
        r_scores.append(recall_at_k(retrieved, relevant, k))
        ndcg_scores.append(ndcg_at_k(retrieved, relevant, k))
        f1_scores.append(f1_at_k(retrieved, relevant, k))

    return {
        "strategy": strategy_name,
        "n": len(examples),
        "k": k,
        f"P@{k}": np.mean(p_scores),
        f"R@{k}": np.mean(r_scores),
        f"NDCG@{k}": np.mean(ndcg_scores),
        f"F1@{k}": np.mean(f1_scores),
    }


def print_results(results: list[dict], k: int) -> None:
    headers = ["Strategy", "N", f"P@{k}", f"R@{k}", f"NDCG@{k}", f"F1@{k}"]
    rows = [
        [
            r["strategy"],
            str(r["n"]),
            f"{r[f'P@{k}']:.4f}",
            f"{r[f'R@{k}']:.4f}",
            f"{r[f'NDCG@{k}']:.4f}",
            f"{r[f'F1@{k}']:.4f}",
        ]
        for r in results
    ]
    col_w = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*row))
    print(sep)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Policy RAG retrieval strategies")
    parser.add_argument("--k", type=int, default=5, help="Cut-off rank K (default: 5)")
    parser.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="all",
        help="Which examples to evaluate (all/train/val, default: all)",
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES) + ["all"],
        default="all",
        help="Strategy to run (default: all)",
    )
    args = parser.parse_args()

    examples = json.loads(LABELED_DATA.read_text(encoding="utf-8"))
    if args.split == "train":
        examples = examples[:80]
    elif args.split == "val":
        examples = examples[80:]

    strategy_names = list(STRATEGIES) if args.strategy == "all" else [args.strategy]

    print(f"\nEvaluating {len(examples)} examples | K={args.k} | split={args.split}\n")

    results = []
    for name in strategy_names:
        print(f"  Running strategy: {name}…")
        result = evaluate(examples, STRATEGIES[name], args.k, name)
        results.append(result)

    print()
    print_results(results, args.k)

    # Save JSON for further analysis
    out_path = Path("eval_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()

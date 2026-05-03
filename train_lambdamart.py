"""
Train a LambdaMART reranker on the 100 labeled compliance examples.

Run AFTER build_policy_rag.py (needs the ChromaDB + BM25 index to exist).

Usage:
    python train_lambdamart.py

Output:
    lambdamart_model.txt   — LightGBM model file loaded automatically by rag_models.py
    lambdamart_train.log   — per-iteration training log
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LABELED_DATA = Path("raw_policy/labeled_data.json")
MODEL_OUT = Path("lambdamart_model.txt")

# 80 / 20 train-val split (first 80 for training, last 20 for validation)
TRAIN_N = 80


def _load_models():
    """Import lazily so the script fails fast if deps are missing."""
    try:
        import lightgbm as lgb  # noqa: F401
    except ImportError:
        sys.exit("lightgbm not installed. Run: pip install lightgbm")
    from rag_models import (
        bm25_retrieve,
        dense_retrieve,
        get_reranker_model,
        rrf_fusion,
    )
    return lgb, bm25_retrieve, dense_retrieve, get_reranker_model, rrf_fusion


def build_features(query: str, chunks, reranker) -> np.ndarray:
    """
    Feature vector per (query, chunk) pair:
      0 — BM25 score (0 if unavailable)
      1 — dense cosine score (0 if unavailable)
      2 — cross-encoder reranker score
      3 — normalised query length
      4 — normalised chunk length
      5 — severity: 1.0 = high, 0.5 = medium
    """
    if not chunks:
        return np.empty((0, 6))

    reranker_scores = reranker.predict([[query, c.text] for c in chunks])
    severity_map = {"high": 1.0, "medium": 0.5}

    rows = []
    for chunk, rscore in zip(chunks, reranker_scores):
        rows.append([
            chunk.bm25_score or 0.0,
            chunk.dense_score or 0.0,
            float(rscore),
            len(query) / 500.0,
            len(chunk.text) / 400.0,
            severity_map.get(chunk.metadata.get("severity", "medium"), 0.5),
        ])
    return np.array(rows, dtype=np.float32)


def build_dataset(examples: list[dict], bm25_retrieve, dense_retrieve, rrf_fusion, reranker):
    """
    For each labeled example retrieve candidates, compute features, assign labels.

    Label scheme:
      2 — chunk's policy_id is in the example's relevant_policy_ids
      0 — otherwise
    """
    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    group_sizes: list[int] = []

    for ex in examples:
        query = ex["query"]
        relevant = set(ex["relevant_policy_ids"])

        dense = dense_retrieve(query, n_results=20)
        bm25 = bm25_retrieve(query, n_results=20)
        candidates = rrf_fusion([dense, bm25])[:20]

        if not candidates:
            continue

        X = build_features(query, candidates, reranker)
        y = np.array(
            [2 if c.metadata.get("policy_id") in relevant else 0 for c in candidates],
            dtype=np.int32,
        )

        all_X.append(X)
        all_y.append(y)
        group_sizes.append(len(candidates))

    return np.vstack(all_X), np.concatenate(all_y), group_sizes


def main():
    lgb, bm25_retrieve, dense_retrieve, get_reranker_model, rrf_fusion = _load_models()

    examples = json.loads(LABELED_DATA.read_text(encoding="utf-8"))
    print(f"Loaded {len(examples)} labeled examples.")

    train_examples = examples[:TRAIN_N]
    val_examples = examples[TRAIN_N:]

    print("Loading retrieval models (this may take a while on first run)…")
    reranker = get_reranker_model()

    print(f"Building training features for {len(train_examples)} queries…")
    X_train, y_train, groups_train = build_dataset(
        train_examples, bm25_retrieve, dense_retrieve, rrf_fusion, reranker
    )

    print(f"Building validation features for {len(val_examples)} queries…")
    X_val, y_val, groups_val = build_dataset(
        val_examples, bm25_retrieve, dense_retrieve, rrf_fusion, reranker
    )

    print(f"Train: {X_train.shape[0]} pairs | Val: {X_val.shape[0]} pairs")

    train_data = lgb.Dataset(X_train, label=y_train, group=groups_train)
    val_data = lgb.Dataset(X_val, label=y_val, group=groups_val, reference=train_data)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [3, 5],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 5,
        "num_iterations": 200,
        "early_stopping_rounds": 30,
        "verbose": -1,
        "label_gain": [0, 1, 2],
    }

    print("Training LambdaMART…")
    callbacks = [
        lgb.log_evaluation(period=10),
        lgb.early_stopping(stopping_rounds=30),
    ]
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=["val"],
        callbacks=callbacks,
    )

    model.save_model(str(MODEL_OUT))
    print(f"\nModel saved to {MODEL_OUT}")
    print(f"Best iteration: {model.best_iteration}")

    # Quick feature importance
    importance = model.feature_importance(importance_type="gain")
    names = ["bm25_score", "dense_score", "reranker_score", "query_len", "chunk_len", "severity"]
    print("\nFeature importance (gain):")
    for name, imp in sorted(zip(names, importance), key=lambda x: -x[1]):
        print(f"  {name:<20} {imp:.1f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "enterprise_policy"
EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
RERANKER_NAME = "BAAI/bge-reranker-large"
GENERATOR_NAME = "Qwen/Qwen2.5-3B-Instruct"
BM25_INDEX_PATH = Path("bm25_index.pkl")
LAMBDAMART_PATH = Path("lambdamart_model.txt")

# BGE requires this prefix when encoding retrieval queries (not documents)
BGE_QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："

_generation_model = None
_generation_tokenizer = None
_embed_model = None
_reranker_model = None
_bm25_data = None
_lambdamart_model = None


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float | None = None
    bm25_score: float | None = None
    dense_score: float | None = None
    reranker_score: float | None = None


# ── Model loaders ────────────────────────────────────────────────────────────

def get_tokenizer():
    global _generation_tokenizer
    if _generation_tokenizer is None:
        _generation_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    return _generation_tokenizer


def get_generation_model():
    global _generation_model
    if _generation_model is None:
        _generation_model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_NAME, torch_dtype="auto", device_map="auto"
        )
    return _generation_model


def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def get_reranker_model():
    global _reranker_model
    if _reranker_model is None:
        _reranker_model = CrossEncoder(RERANKER_NAME)
    return _reranker_model


def get_bm25_data() -> dict | None:
    global _bm25_data
    if _bm25_data is None and BM25_INDEX_PATH.exists():
        with open(BM25_INDEX_PATH, "rb") as f:
            _bm25_data = pickle.load(f)
    return _bm25_data


def get_lambdamart_model():
    global _lambdamart_model
    if _lambdamart_model is None and LAMBDAMART_PATH.exists():
        import lightgbm as lgb
        _lambdamart_model = lgb.Booster(model_file=str(LAMBDAMART_PATH))
    return _lambdamart_model


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


# ── Generation ───────────────────────────────────────────────────────────────

def ask_qwen(question: str, system_prompt: str | None = None, max_new_tokens: int = 512) -> str:
    tokenizer = get_tokenizer()
    model = get_generation_model()
    messages = [
        {"role": "system", "content": system_prompt or "你是一个中国企业合规审核助手。"},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs, max_new_tokens=max_new_tokens,
            do_sample=False, temperature=None, top_p=None,
        )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


# ── Query expansion ───────────────────────────────────────────────────────────

def expand_query(document_text: str) -> List[str]:
    """Ask Qwen to generate 3 focused sub-queries from the document."""
    prompt = (
        "请从以下文档中提取3个最关键的合规审查问题，每个问题单独一行，不加编号，"
        "问题应聚焦具体的合规风险点。\n\n"
        f"文档片段：\n{document_text[:1500]}\n\n"
        "输出3个合规问题（每行一个）："
    )
    system = "你是一个中国企业合规专家，擅长识别文档中的合规风险。"
    response = ask_qwen(prompt, system_prompt=system, max_new_tokens=150)
    return [q.strip() for q in response.strip().splitlines() if q.strip()][:3]


# ── Retrieval ────────────────────────────────────────────────────────────────

def _tokenize_chinese(text: str) -> List[str]:
    import jieba
    return list(jieba.cut(text))


def dense_retrieve(query: str, n_results: int = 15) -> List[RetrievedChunk]:
    collection = get_collection()
    embed_model = get_embed_model()
    query_embedding = embed_model.encode(
        [BGE_QUERY_PREFIX + query], normalize_embeddings=True
    ).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results.get("distances", [[]])[0],
    ):
        score = float(1.0 - dist)
        chunks.append(RetrievedChunk(text=doc, metadata=meta or {}, dense_score=score, score=score))
    return chunks


def bm25_retrieve(query: str, n_results: int = 15) -> List[RetrievedChunk]:
    data = get_bm25_data()
    if data is None:
        return []
    import numpy as np
    bm25 = data["bm25"]
    documents = data["documents"]
    metadatas = data["metadatas"]
    scores = bm25.get_scores(_tokenize_chinese(query))
    top_indices = np.argsort(scores)[::-1][:n_results]
    return [
        RetrievedChunk(
            text=documents[i],
            metadata=metadatas[i],
            bm25_score=float(scores[i]),
            score=float(scores[i]),
        )
        for i in top_indices
    ]


def rrf_fusion(
    lists: List[List[RetrievedChunk]], k: int = 60
) -> List[RetrievedChunk]:
    """Reciprocal Rank Fusion across any number of ranked lists."""
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, RetrievedChunk] = {}
    for ranked_list in lists:
        for rank, chunk in enumerate(ranked_list):
            key = chunk.text
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = chunk
            else:
                # Merge scores onto existing entry
                if chunk.bm25_score is not None:
                    chunk_map[key].bm25_score = chunk.bm25_score
                if chunk.dense_score is not None:
                    chunk_map[key].dense_score = chunk.dense_score
    merged = sorted(chunk_map.keys(), key=lambda k: rrf_scores[k], reverse=True)
    for key in merged:
        chunk_map[key].score = rrf_scores[key]
    return [chunk_map[k] for k in merged]


def retrieve_chunks(query: str, n_results: int = 15) -> List[RetrievedChunk]:
    """Hybrid BM25 + dense retrieval fused with RRF."""
    dense = dense_retrieve(query, n_results=n_results)
    bm25 = bm25_retrieve(query, n_results=n_results)
    return rrf_fusion([dense, bm25])


# ── Reranking ─────────────────────────────────────────────────────────────────

def rerank_chunks(query: str, chunks: Sequence[RetrievedChunk], top_n: int = 5) -> List[RetrievedChunk]:
    if not chunks:
        return []
    reranker = get_reranker_model()
    scores = reranker.predict([[query, c.text] for c in chunks])
    ranked = sorted(
        zip(chunks, scores), key=lambda x: float(x[1]), reverse=True
    )
    result = []
    for chunk, s in ranked[:top_n]:
        chunk.reranker_score = float(s)
        chunk.score = float(s)
        result.append(chunk)
    return result


def lambdamart_rerank(query: str, chunks: List[RetrievedChunk], top_n: int = 5) -> List[RetrievedChunk]:
    """Rerank with trained LambdaMART model when available."""
    model = get_lambdamart_model()
    if model is None or not chunks:
        return chunks[:top_n]
    import numpy as np
    X = np.array([
        [
            c.bm25_score or 0.0,
            c.dense_score or 0.0,
            c.reranker_score or 0.0,
            len(query) / 500.0,
            len(c.text) / 400.0,
        ]
        for c in chunks
    ])
    lm_scores = model.predict(X)
    ranked = sorted(zip(chunks, lm_scores), key=lambda x: float(x[1]), reverse=True)
    result = []
    for chunk, s in ranked[:top_n]:
        chunk.score = float(s)
        result.append(chunk)
    return result


# ── Main audit pipeline ───────────────────────────────────────────────────────

def audit_document(
    document_text: str,
    use_reranker: bool = True,
    use_query_expansion: bool = True,
) -> Dict[str, Any]:
    # Step 1: build query list (original + expanded sub-queries)
    queries: List[str] = [document_text[:3000]]
    if use_query_expansion:
        try:
            queries.extend(expand_query(document_text))
        except Exception:
            pass

    # Step 2: multi-query retrieval — retrieve for each query then fuse
    per_query_results = [retrieve_chunks(q, n_results=10) for q in queries]
    fused = rrf_fusion(per_query_results)[:10]

    # Step 3: cross-encoder reranking
    primary_query = document_text[:3000]
    if use_reranker:
        chunks = rerank_chunks(primary_query, fused, top_n=5)
        chunks = lambdamart_rerank(primary_query, chunks, top_n=5)
    else:
        chunks = fused[:5]

    # Step 4: build context and generate
    context = "\n\n".join(
        f"Policy ID: {c.metadata.get('policy_id')}\n"
        f"Category: {c.metadata.get('category')}\n"
        f"Severity: {c.metadata.get('severity')}\n"
        f"Title: {c.metadata.get('title')}\n"
        f"Content: {c.text}"
        for c in chunks
    )
    system_prompt = (
        "你是中国企业政策合规审核助手。"
        "你必须只基于提供的 Policy Context 进行审核。"
        "输出中文，结构清晰，不要编造不存在的政策。"
    )
    user_prompt = f"""Policy Context:
{context}

待审核文档：
{document_text[:5000]}

请输出：
1. 总体风险等级：Low / Medium / High
2. 主要违规点
3. 命中的政策 ID 和原因
4. 修改建议
5. 可直接替换的合规改写示例
"""
    answer = ask_qwen(user_prompt, system_prompt=system_prompt, max_new_tokens=700)
    return {
        "model": "Qwen/Qwen2.5-3B-Instruct + BGE + BM25 + LambdaMART",
        "answer": answer,
        "sub_queries": queries[1:],
        "chunks": [
            {
                "metadata": c.metadata,
                "score": c.score,
                "bm25_score": c.bm25_score,
                "dense_score": c.dense_score,
                "reranker_score": c.reranker_score,
                "text": c.text,
            }
            for c in chunks
        ],
    }

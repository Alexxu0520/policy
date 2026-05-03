from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import chromadb
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "enterprise_policy"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GENERATOR_NAME = "Qwen/Qwen2.5-3B-Instruct"


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float | None = None


_generation_model = None
_generation_tokenizer = None
_embed_model = None
_reranker_model = None


def get_tokenizer():
    global _generation_tokenizer
    if _generation_tokenizer is None:
        _generation_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_NAME)
    return _generation_tokenizer


def get_generation_model():
    global _generation_model
    if _generation_model is None:
        _generation_model = AutoModelForCausalLM.from_pretrained(
            GENERATOR_NAME,
            torch_dtype="auto",
            device_map="auto",
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


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_collection(name=COLLECTION_NAME)


def ask_qwen(question: str, system_prompt: str | None = None, max_new_tokens: int = 512) -> str:
    tokenizer = get_tokenizer()
    model = get_generation_model()

    messages = [
        {
            "role": "system",
            "content": system_prompt or "你是一个中国企业合规审核助手。",
        },
        {"role": "user", "content": question},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def retrieve_chunks(query: str, n_results: int = 8) -> List[RetrievedChunk]:
    collection = get_collection()
    embed_model = get_embed_model()
    query_embedding = embed_model.encode([query]).tolist()[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results.get("distances", [[]])[0]

    chunks: List[RetrievedChunk] = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        dist = distances[i] if i < len(distances) else None
        score = None if dist is None else float(-dist)
        chunks.append(RetrievedChunk(text=doc, metadata=meta or {}, score=score))
    return chunks


def rerank_chunks(query: str, chunks: Sequence[RetrievedChunk], top_n: int = 4) -> List[RetrievedChunk]:
    if not chunks:
        return []
    reranker = get_reranker_model()
    pairs = [[query, c.text] for c in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(
        [RetrievedChunk(text=c.text, metadata=c.metadata, score=float(s)) for c, s in zip(chunks, scores)],
        key=lambda c: c.score if c.score is not None else float("-inf"),
        reverse=True,
    )
    return ranked[:top_n]


def audit_document(document_text: str, use_reranker: bool = True) -> Dict[str, Any]:
    query = document_text[:3000]
    retrieved = retrieve_chunks(query, n_results=10)
    chunks = rerank_chunks(query, retrieved, top_n=5) if use_reranker else retrieved[:5]

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

    user_prompt = f"""
Policy Context:
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
        "model": "Qwen/Qwen2.5-3B-Instruct + RAG",
        "answer": answer,
        "chunks": [
            {"metadata": c.metadata, "score": c.score, "text": c.text}
            for c in chunks
        ],
    }

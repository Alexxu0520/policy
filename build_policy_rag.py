import json
import pickle
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from rag_models import (
    BM25_INDEX_PATH,
    BGE_QUERY_PREFIX,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
)

POLICY_JSON = Path("raw_policy/policy_knowledge.json")

_SENT_SPLIT = re.compile(r"(?<=[。！？；])")


def chinese_sentence_chunk(text: str, max_chars: int = 400, overlap_sents: int = 1) -> list[str]:
    """Split on Chinese sentence boundaries; overlap by keeping the last sentence."""
    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    if not sentences:
        return [text]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append("".join(current))
            current = current[-overlap_sents:]
            current_len = sum(len(s) for s in current)
        current.append(sent)
        current_len += len(sent)

    if current:
        chunks.append("".join(current))
    return chunks


def _tokenize_chinese(text: str) -> list[str]:
    import jieba
    return list(jieba.cut(text))


def main():
    policies = json.loads(POLICY_JSON.read_text(encoding="utf-8"))
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    documents, metadatas, ids = [], [], []
    for p in policies:
        full_text = f"{p['title']}。{p['text']}"
        for idx, chunk in enumerate(chinese_sentence_chunk(full_text)):
            documents.append(chunk)
            metadatas.append({
                "policy_id": p["id"],
                "category": p["category"],
                "severity": p["severity"],
                "title": p["title"],
                "chunk_index": idx,
            })
            ids.append(f"{p['id']}-{idx}")

    # BGE encodes documents without the query prefix
    embeddings = embed_model.encode(
        documents, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).tolist()
    collection.add(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
    print(f"Built ChromaDB collection '{COLLECTION_NAME}' with {len(documents)} chunks.")

    # Build and persist BM25 index alongside ChromaDB
    tokenized = [_tokenize_chinese(doc) for doc in documents]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "documents": documents, "metadatas": metadatas, "ids": ids}, f)
    print(f"BM25 index saved to {BM25_INDEX_PATH} ({len(documents)} entries).")


if __name__ == "__main__":
    main()

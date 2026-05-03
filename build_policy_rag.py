import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from rag_models import CHROMA_DIR, COLLECTION_NAME, EMBED_MODEL_NAME

POLICY_JSON = Path("raw_policy/policy_knowledge.json")


def chunk_text(text: str, chunk_size: int = 450, overlap: int = 80):
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


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
        for idx, chunk in enumerate(chunk_text(full_text)):
            documents.append(chunk)
            metadatas.append({
                "policy_id": p["id"],
                "category": p["category"],
                "severity": p["severity"],
                "title": p["title"],
                "chunk_index": idx,
            })
            ids.append(f"{p['id']}-{idx}")

    embeddings = embed_model.encode(documents, batch_size=32, show_progress_bar=True).tolist()
    collection.add(documents=documents, metadatas=metadatas, embeddings=embeddings, ids=ids)
    print(f"Built Chroma collection '{COLLECTION_NAME}' with {len(documents)} policy chunks.")


if __name__ == "__main__":
    main()

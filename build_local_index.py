import os, pickle, re
import fitz
import numpy as np
from openai import OpenAI

PDF_PATH = "machine-learning.pdf"
KEY_PATH = "key.txt"
CACHE_PATH = "local_vectorstore.pkl"

CHUNK_SIZE = 1800
OVERLAP = 150
EMBED_MODEL = "text-embedding-3-small"

def load_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()

def clean_text(s: str) -> str:
    s = s.replace("\n", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def chunk_text(text: str, chunk_size: int, overlap: int):
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks

def embed_batch(client: OpenAI, texts, model=EMBED_MODEL):
    resp = client.embeddings.create(model=model, input=texts)
    return [d.embedding for d in resp.data]

def main():
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"Missing {PDF_PATH} in current folder.")
    api_key = load_key(KEY_PATH)
    client = OpenAI(api_key=api_key)

    doc = fitz.open(PDF_PATH)

    all_chunks = []
    for page_i in range(len(doc)):
        page_text = clean_text(doc[page_i].get_text("text"))
        if not page_text:
            continue
        chunks = chunk_text(page_text, CHUNK_SIZE, OVERLAP)
        for c in chunks:
            all_chunks.append({"text": c, "page_number": page_i + 1})

    print(f"Total chunks: {len(all_chunks)}")

    vectors = []
    batch_size = 64
    for i in range(0, len(all_chunks), batch_size):
        batch_texts = [x["text"] for x in all_chunks[i:i+batch_size]]
        embs = embed_batch(client, batch_texts)
        vectors.extend(embs)
        print(f"Embedded {min(i+batch_size, len(all_chunks))}/{len(all_chunks)}")

    vectors = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors = vectors / norms

    payload = {"chunks": all_chunks, "vectors": vectors}
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved local vectorstore to: {CACHE_PATH}")

if __name__ == "__main__":
    main()
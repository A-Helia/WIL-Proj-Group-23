"""
Builds a Pyserini-compatible Faiss dense index with SentenceTransformers
(all-MiniLM-L6-v2), using the model's own pooling and normalised vectors.
"""
import os
import shutil
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_DIR = "../data"
COLLECTION = f"{DATA_DIR}/collection.csv"
EMBEDDINGS_DIR = "../target/indexes/dense_pyserini"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_faiss_index_direct():
    if os.path.exists(EMBEDDINGS_DIR):
        shutil.rmtree(EMBEDDINGS_DIR)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    df = pd.read_csv(COLLECTION)
    print(f"Encoding {len(df)} passages on CPU...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    # normalised vectors, so inner product equals cosine similarity
    embeddings = model.encode(df["passage"].tolist(), normalize_embeddings=True).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "index"))

    # both file names, to satisfy all Pyserini versions
    for filename in ["docid", "docids"]:
        with open(os.path.join(EMBEDDINGS_DIR, filename), "w") as f:
            f.writelines(f"{d}\n" for d in df["passage_id"])

    print("Wrote dense index")

if __name__ == "__main__":
    build_faiss_index_direct()
"""
Builds a native Pyserini-compatible Faiss dense index using 
SentenceTransformers (all-MiniLM-L6-v2) directly to bypass Pyserini caching bugs.
"""
import os
import shutil
import numpy as np
import pandas as pd
import faiss

DATA_DIR = "../data"
COLLECTION = f"{DATA_DIR}/collection.csv"
EMBEDDINGS_DIR = "../target/indexes/dense_pyserini"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def build_faiss_index_direct():
    print(f"Loading {MODEL_NAME} via Transformers pipeline...")
    from transformers import pipeline
    
    if os.path.exists(EMBEDDINGS_DIR):
        shutil.rmtree(EMBEDDINGS_DIR)
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    df = pd.read_csv(COLLECTION)
    passages = df["passage"].tolist()
    docids = df["passage_id"].tolist()

    print(f"Generating vectors for {len(passages)} items using CPU...")
    encoder = pipeline("feature-extraction", model=MODEL_NAME, device=-1)
    
    vectors = []
    for passage in passages:
        res = encoder(passage)
        vec = np.mean(res[0], axis=0)  # Extract structural pooled token embeddings safely
        vectors.append(vec)
        
    embeddings = np.array(vectors).astype('float32')
    dimension = embeddings.shape[1]

    print(f"Compiling vector architecture with dimension size: {dimension}")
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, "index"))
    
    # Save both plural and singular text maps to satisfy all Pyserini versions
    for filename in ["docid", "docids"]:
        with open(os.path.join(EMBEDDINGS_DIR, filename), "w") as f:
            for docid in docids:
                f.write(f"{docid}\n")
            
    print("Successfully wrote a clean MiniLM vector index layout!")

if __name__ == "__main__":
    build_faiss_index_direct()

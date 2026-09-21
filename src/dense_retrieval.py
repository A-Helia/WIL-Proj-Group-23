"""
Dense retrieval search script using Pyserini's native FaissSearcher.
Requires a dense FAISS index built first with build_dense_index_pyserini.py.
"""
import os
import pandas as pd
from pyserini.search.faiss import FaissSearcher

os.environ.setdefault("OPENAI_API_KEY", "unused")

DATA_DIR = "../data"
RUNS_DIR = "../target/runs"
INDEX_DIR = "../target/indexes/dense_pyserini"
TOPICS = f"{DATA_DIR}/topics.csv"
OUTPUT_PATH = f"{RUNS_DIR}/dense.txt"
RUN_TAG = "gymrag.dense"
QUERY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
NUM_HITS = 20

def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    topics = pd.read_csv(TOPICS)
    
    print(f"Initializing FaissSearcher with encoder: {QUERY_ENCODER}...")
    searcher = FaissSearcher(INDEX_DIR, QUERY_ENCODER)

    print(f"Running dense search for {len(topics)} questions...")
    with open(OUTPUT_PATH, "w") as f:
        for question_id, question in topics[["question_id", "question"]].values:
            hits = searcher.search(question, NUM_HITS)
            for rank, hit in enumerate(hits, start=1):
                f.write(f"{question_id} Q0 {hit.docid} {rank} {hit.score:.4f} {RUN_TAG}\n")

    print(f"Wrote dense run file to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

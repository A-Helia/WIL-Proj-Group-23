"""
Dense retrieval using Pyserini's own FaissSearcher — the exact class
Walert's search.py uses for its dense (DPR + FAISS) run. Requires the
index built first with build_dense_index_pyserini.py.

Downloads the DPR question encoder (~440MB) on first run.

Usage:
    python build_dense_index_pyserini.py    # once, or whenever collection.csv changes
    python dense_retrieval_pyserini.py
"""
import os
os.environ.setdefault("OPENAI_API_KEY", "unused")

import pandas as pd
from pyserini.search.faiss import FaissSearcher

DATA_DIR = "../data"
RUNS_DIR = "../target/runs"
INDEX_DIR = "../target/indexes/dense_pyserini"
TOPICS = f"{DATA_DIR}/topics.csv"
OUTPUT_PATH = f"{RUNS_DIR}/dense.txt"
RUN_TAG = "gymrag.dense.faiss"
NUM_HITS = 20
QUERY_ENCODER = "facebook/dpr-question_encoder-multiset-base"


def main():
    topics = pd.read_csv(TOPICS)
    searcher = FaissSearcher(INDEX_DIR, QUERY_ENCODER)

    with open(OUTPUT_PATH, "w") as f:
        for question_id, question in topics[["question_id", "question"]].values:
            hits = searcher.search(question, NUM_HITS)
            for rank, hit in enumerate(hits, start=1):
                f.write(f"{question_id} Q0 {hit.docid} {rank} {hit.score:.4f} {RUN_TAG}\n")

    print(f"Wrote run file for {len(topics)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
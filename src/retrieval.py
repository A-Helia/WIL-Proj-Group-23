"""
Equivalent of Walert's search.py — this version uses the SAME retrieval
engine Walert uses: Pyserini's LuceneSearcher (Anserini/Lucene BM25),
not a Python reimplementation. Requires a Lucene index built first with
build_index.py.

Usage:
    python build_index.py     # once, or whenever collection.csv changes
    python retrieval.py
"""
import pandas as pd
from pyserini.search.lucene import LuceneSearcher

DATA_DIR = "../data"
RUNS_DIR = "../target/runs"
INDEX_DIR = "../target/indexes/bm25"
TOPICS = f"{DATA_DIR}/topics.csv"
OUTPUT_PATH = f"{RUNS_DIR}/bm25.txt"
RUN_TAG = "gymrag.bm25"
NUM_HITS = 20


def main():
    topics = pd.read_csv(TOPICS)
    searcher = LuceneSearcher(INDEX_DIR)

    with open(OUTPUT_PATH, "w") as f:
        for question_id, question in topics[["question_id", "question"]].values:
            hits = searcher.search(question, NUM_HITS)
            for rank, hit in enumerate(hits, start=1):
                f.write(f"{question_id} Q0 {hit.docid} {rank} {hit.score:.4f} {RUN_TAG}\n")

    print(f"Wrote run file for {len(topics)} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

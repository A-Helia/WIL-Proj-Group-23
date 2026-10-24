"""Writes the RRF run file so custom_eval.py can score it with bm25 and dense."""
import os
os.environ.setdefault("OPENAI_API_KEY", "unused")  # pyserini import quirk
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
from pyserini.encode import AutoQueryEncoder

DATA_DIR = "../data"
BM25_DIR = "../target/indexes/bm25"
DENSE_DIR = "../target/indexes/dense_pyserini"
OUTPUT_PATH = "../target/runs/rrf.txt"
QUERY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
DEPTH = 6      # hits per engine, same as chatbot.py (TOP_K * 2)
RRF_K = 50     # same constant as chatbot.py
RUN_TAG = "gymrag.rrf"

def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    topics = pd.read_csv(f"{DATA_DIR}/topics.csv")
    bm25 = LuceneSearcher(BM25_DIR)
    bm25.set_bm25(k1=0.8, b=0.4)
    # encoder settings must match chatbot.py and the index builder
    encoder = AutoQueryEncoder(QUERY_ENCODER, device="cpu", pooling="mean", l2_norm=True)
    dense = FaissSearcher(DENSE_DIR, encoder)

    with open(OUTPUT_PATH, "w") as f:
        for qid, question in topics[["question_id", "question"]].values:
            scores = {}
            for hits in (bm25.search(question, DEPTH), dense.search(question, DEPTH)):
                for rank, hit in enumerate(hits, start=1):
                    scores[hit.docid] = scores.get(hit.docid, 0.0) + 1.0 / (RRF_K + rank)
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {RUN_TAG}\n")
    print(f"Wrote {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
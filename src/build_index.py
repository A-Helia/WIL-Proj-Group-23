"""
Builds the Lucene/Anserini index from collection.csv, matching what
Walert's index.sh does before running search.py. Run this once, and
again any time collection.csv changes.

Usage:
    python build_index.py
"""
import json
import os
import subprocess
import pandas as pd

# Same Pyserini import-time quirk as in retrieval.py/chatbot.py — set a
# harmless dummy value so the subprocess below never hits the missing
# OPENAI_API_KEY crash, even if a future pyserini version touches it here.
os.environ.setdefault("OPENAI_API_KEY", "unused")

DATA_DIR = "../data"
COLLECTION = f"{DATA_DIR}/collection.csv"
JSONL_DIR = f"{DATA_DIR}/collection_jsonl"
INDEX_DIR = "../target/indexes/bm25"


def collection_to_jsonl():
    os.makedirs(JSONL_DIR, exist_ok=True)
    df = pd.read_csv(COLLECTION)
    out_path = os.path.join(JSONL_DIR, "docs.jsonl")
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"id": row["passage_id"], "contents": row["passage"]}) + "\n")
    print(f"Wrote {len(df)} docs to {out_path}")


def build_lucene_index():
    cmd = [
        "python3", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", JSONL_DIR,
        "--index", INDEX_DIR,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    collection_to_jsonl()
    build_lucene_index()
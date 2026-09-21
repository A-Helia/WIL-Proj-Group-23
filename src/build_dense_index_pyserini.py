"""
Builds a dense/FAISS index using Pyserini's OWN encoding + indexing tool
(pyserini.encode) — the actual path Walert's search.py assumes exists,
via FaissSearcher. Unlike build_dense_index.py (which hand-rolls the
encoding with raw transformers + faiss calls), this produces an index in
Pyserini's native format, so FaissSearcher can load it directly, same as
Walert's own code does.

Downloads the DPR context encoder (~440MB) from Hugging Face on first
run. Run once, and again any time collection.csv changes.

Usage:
    python build_dense_index_pyserini.py
"""
import json
import os
import shutil
import subprocess
import pandas as pd

# Same pyserini import-time quirk as the BM25 scripts — harmless dummy value.
os.environ.setdefault("OPENAI_API_KEY", "unused")

DATA_DIR = "../data"
COLLECTION = f"{DATA_DIR}/collection.csv"
# Separate JSONL dir from build_index.py's — Lucene's JsonCollection
# expects a "contents" field, but pyserini.encode's own corpus loader
# expects a field literally named "text", regardless of the --fields
# value passed on the command line. Using one shared JSONL for both
# tools is what caused the KeyError: 'text' crash.
JSONL_DIR = f"{DATA_DIR}/collection_jsonl_dense"
EMBEDDINGS_DIR = "../target/indexes/dense_pyserini"
CTX_ENCODER = "facebook/dpr-ctx_encoder-multiset-base"


def collection_to_jsonl():
    os.makedirs(JSONL_DIR, exist_ok=True)
    df = pd.read_csv(COLLECTION)
    out_path = os.path.join(JSONL_DIR, "docs.jsonl")
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps({"id": row["passage_id"], "text": row["passage"]}) + "\n")
    print(f"Wrote {len(df)} docs to {out_path}")


def build_faiss_index():
    # remove any stale/incomplete index from a previous failed run — a
    # partial index directory can otherwise get loaded successfully by
    # FaissSearcher later and silently return zero hits for everything
    if os.path.exists(EMBEDDINGS_DIR):
        shutil.rmtree(EMBEDDINGS_DIR)

    cmd = [
        "python3", "-m", "pyserini.encode",
        "input", "--corpus", JSONL_DIR, "--fields", "text",
        "output", "--embeddings", EMBEDDINGS_DIR, "--to-faiss",
        "encoder", "--encoder", CTX_ENCODER, "--encoder-class", "dpr",
        "--fields", "text", "--batch-size", "8", "--device", "cpu",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    collection_to_jsonl()
    build_faiss_index()
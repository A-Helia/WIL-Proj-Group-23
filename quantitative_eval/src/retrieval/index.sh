ENCODER="tct_colbert-v2-hnp-msmarco"
EMBEDDINGS="../../target/embeddings/$ENCODER"

INDEX="../../target/indexes/$ENCODER-faiss"
python -m pyserini.index.faiss \
  --input $EMBEDDINGS \
  --output $INDEX \
  --hnsw
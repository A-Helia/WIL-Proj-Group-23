
#encode
CORPUS="../../data/collection.jsonl"
ENCODER="tct_colbert-v2-hnp-msmarco"
EMBEDDINGS="../../target/embeddings/$ENCODER"

python -m pyserini.encode \
 input   --corpus $CORPUS  \
          --fields text \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings $EMBEDDINGS \
            --to-faiss \
  encoder --encoder castorini/$ENCODER \
          --fields text \
          --batch 32 \
          --device mps 
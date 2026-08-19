CORPUS="../../data/collection"
INDEX="../../target/indexes/bm25"

 python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $CORPUS \
  --language en \
  --index $INDEX \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw 
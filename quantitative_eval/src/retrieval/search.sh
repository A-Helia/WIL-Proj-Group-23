#DEPRECATED
#It almost worked! Unfortunately, the topics parameter reads a hardcoded set of topics, and not a file with new topics.
python -m pyserini.search.faiss \
  --index ../../target/index \
  --encoder facebook/dpr-question_encoder-multiset-base \
  --topics msmarco-passage-dev-subset \
  --topics-format default \
  --output ../../target/runs/run.faiss.txt \
  --output-format trec \
  --hits 100
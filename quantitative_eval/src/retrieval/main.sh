##1. Parse test collection:
#python data.py

##2. Encode test collection:
#./encode.sh

##3. Index test collection:
#./index.sh

##4. Run the searches:
#python search.py

##5. Evaluate the runs:
#python eval.py  known ../../data/qrels.txt ../../target/runs/walert-intent.txt ../../target/runs/rag-bm25.txt ../../target/runs/rag-dense-faiss.txt
python eval.py  inferred ../../data/qrels.txt ../../target/runs/walert-intent.txt ../../target/runs/rag-bm25.txt ../../target/runs/rag-dense-faiss.txt
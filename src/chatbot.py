"""
Hybrid FAQ chatbot: BM25 Sparse Search + MiniLM Dense Search via Reciprocal Rank Fusion (RRF).
Combines exact keyword matching with semantic vector meanings to feed the local LLM.

Setup:
    pip install -r ../requirements.txt

Usage:
    python chatbot.py
    python chatbot.py "How much protein do I need for muscle gain?"
"""
import logging
import os
import sys
import pandas as pd
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.faiss import FaissSearcher
# Import the encoder class from the correct pyserini.encode module
from pyserini.encode import AutoQueryEncoder
from transformers import pipeline
import warnings

# 1. Quiet down the Transformers warning logger engine
logging.getLogger("transformers").setLevel(logging.ERROR)

# 2. Tell the system to hide standard environment alert logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Mute standard Python warnings and Hugging Face log alerts
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OPENAI_API_KEY", "unused")

DATA_DIR = "../data"
COLLECTION = f"{DATA_DIR}/collection.csv"
BM25_INDEX_DIR = "../target/indexes/bm25"
DENSE_INDEX_DIR = "../target/indexes/dense_pyserini"
TOP_K = 5 # read only the 5 best matching passage

# Aligned with your high-performing 86.6% dense retrieval script
QUERY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a short, direct FAQ bot. Answer the question using ONLY the provided facts. "
    "Do not use markdown headers, do not use bullet points, and do not say 'Based on the text'. "
    "Write exactly one sentence answering how much protein is needed, and exactly one sentence "
    "answering how to split it. End your sentences with inline citations like [1]."
)



class HybridFaqChatbot:
    def __init__(self, collection_path=COLLECTION, bm25_dir=BM25_INDEX_DIR, dense_dir=DENSE_INDEX_DIR, model_name=MODEL_NAME):
        print("Initializing database maps...")
        self.collection = pd.read_csv(collection_path).set_index("passage_id")
        
        print("Loading BM25 sparse keyword index...")
        self.bm25_searcher = LuceneSearcher(bm25_dir)
        self.bm25_searcher.set_bm25(k1=0.8, b=0.4) # Tuned short FAQ document matching parameters
        
        print(f"Loading persistent MiniLM query encoder instance: {QUERY_ENCODER}...")
        # Instantiating the query encoder once at boot time to stop slow query-time reloads
        self.encoder = AutoQueryEncoder(QUERY_ENCODER, device="cpu")
        self.dense_searcher = FaissSearcher(dense_dir, self.encoder)
        
        print(f"Loading local generation model '{model_name}' (runs fully offline on CPU)...")
        self.generator = pipeline("text-generation", model=model_name, device_map="cpu")

    def retrieve_hybrid_rrf(self, question, k=TOP_K):
        # Fetch an extended pool from both engines to find overlapping patterns
        bm25_hits = self.bm25_searcher.search(question, k * 2)
        dense_hits = self.dense_searcher.search(question, k * 2)
        
        rrf_scores = {}
        constant_penalty = 50


        # ---RECIPROCAL RANK FUSION(RRF) ---
        # Instead of guessing whether BM25 or Dense is better, RRF scores documents based on their position in both.
        # Formula applied: Score = 1 / (60 + BM25_Rank) + 1 / (60 + Dense_Rank)
        # If a document ranks 1st in BM25 and 2nd in Dense, its combined score rises, ensuring that documents
        # trusted by both exact keyword matches and semantic meaning maps are prioritized for the LLM context.
        
        
        # Calculate scores for BM25 positions
        for rank, hit in enumerate(bm25_hits, start=1):
            if hit.docid in self.collection.index:
                rrf_scores[hit.docid] = rrf_scores.get(hit.docid, 0.0) + (1.0 / (constant_penalty + rank))
                
        # Calculate and blend scores for Dense positions
        for rank, hit in enumerate(dense_hits, start=1):
            if hit.docid in self.collection.index:
                rrf_scores[hit.docid] = rrf_scores.get(hit.docid, 0.0) + (1.0 / (constant_penalty + rank))
                
        # Sort documents by their combined RRF metric score values
        sorted_docs = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        
        # Extract metadata structures for the top validated documents
        results = []
        for docid, score in sorted_docs[:k]:
            results.append((docid, self.collection.loc[docid]))
            
        return results

    def answer(self, question):
        passages = self.retrieve_hybrid_rrf(question)
        if not passages:
            return "I couldn't find anything in the knowledge base for that yet.", []

        context = "\n".join(
            f"[{i+1}] ({p.source}) {p.passage}" for i, (docid, p) in enumerate(passages)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ]
        # Restricting token limit ensures quick execution times on CPU devices
        output = self.generator(
            messages,
            min_new_tokens=20,
            max_new_tokens=200,
            temperature=0.1, # lower temperatue makes AI more determinitic and stricter
            do_sample=False, # disable sampling to maximize cpu calculation
            eos_token_id=self.generator.tokenizer.eos_token_id, 
            max_length=None
        )

        # FIXED INDEX ACCESS LAYER STRATEGY
        reply = output[0]["generated_text"][-1]["content"]
        return reply, passages


def main():
    bot = HybridFaqChatbot()

    print("\nGym FAQ Hybrid Chatbot active. Type a question, or 'quit' to exit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer, evidence = bot.answer(question)
        print(f"\nAnswer:\n{answer}\n")
        print("Sources Used:")
        for i, (docid, p) in enumerate(evidence, 1):
            print(f"  [{i}] {docid} ({p.source})")


if __name__ == "__main__":
    main()

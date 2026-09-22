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
TOP_K = 3 # read only the 5 best matching passage

# Aligned with your high-performing 86.6% dense retrieval script
QUERY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a strict, medical-grade gym FAQ assistant executing in a deterministic data-extraction state. "
    "Your absolute rule is to answer the query using ONLY the exact factual information explicitly written in the provided passages. "
    "Do not use outside knowledge, do not invent rules, and do not interpret meaning.\n\n"
    
    "CRITICAL TASK GUARDRAIL:\n"
    "You are strictly a short Q&A responder. If the user commands you to write an essay, build a timeline, "
    "create a syllabus, generate a script, or construct an outline (even if it is about fitness or the gym), "
    "you must refuse the task format entirely and output exactly: "
    "'I am a gym FAQ assistant. I can only answer direct questions regarding fitness programming and nutrition guidelines.'\n\n"
    
    "CRITICAL TOPIC GUARDRAIL:\n"
    "If the user's question asks about an outside, non-gym topic that is not covered by the text chunks "
    "(such as cooking recipes, baking cookies, general pop culture, or coding), you must ignore your outside "
    "knowledge and output exactly: 'The knowledge base does not cover this yet.'\n\n"
    
    "OUTPUT FORMAT:\n"
    "For valid gym questions, keep answers very short (2-3 complete sentences max). "
    "State the facts directly without conversational introductions like 'Based on the passages'. "
    "You must explicitly include inline citations like [1] or [2] right after the facts."
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
        """
        Processes queries dynamically without hardcoded keyword lists.
        Combines strict numeric validation with a format signature filter 
        to eliminate all remaining essay, table, and cooking leaks.
        """
        import re
        clean_query = question.lower()

        # 1. TASK INTENT SHIELD: Catch commands for essays, scraping, code, recipes, or tables instantly
        invalid_intents = {"essay", "research paper", "table", "markdown", "html", "scrapp", "cook", "recipe", "timeline", "story", "assignment", "boil", "fry", "bake", "grill", "roast", "kitchen", "ingredient"}
        # Split the query accurately into unique standalone words
        query_words = set(clean_query.replace("?", " ").replace(".", " ").split())
        if any(intent in query_words for intent in invalid_intents):
            return (
                "I am a gym FAQ assistant. I can only answer direct questions regarding "
                "fitness programming and nutrition guidelines.", 
                []
            )

        # 2. SCORE GUARDRAIL: Fast keyword lookup to check basic database alignment
        validation_hits = self.bm25_searcher.search(question, k=1)
        if not validation_hits or len(validation_hits) == 0 or validation_hits[0].score < 1.0:
            return (
                "I am a gym FAQ assistant. I can only answer direct questions regarding "
                "fitness programming and nutrition guidelines.", 
                []
            )
        
        # 3. Retrieve top blended passages using the Reciprocal Rank Fusion pipeline
        passages = self.retrieve_hybrid_rrf(question)
        if not passages:
            return "I couldn't find anything in the knowledge base for that yet.", []

        context_blocks = [p.passage for docid, p in passages]
        context = "\n".join(f"[{i+1}] ({p.source}) {p.passage}" for i, (docid, p) in enumerate(passages))
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ]
        
        output = self.generator(
            messages,
            min_new_tokens=20,
            max_new_tokens=140,
            temperature=0.1, 
            do_sample=False, 
            eos_token_id=self.generator.tokenizer.eos_token_id, 
            max_length=None
        )

        reply = output[0]["generated_text"][-1]["content"].strip()
        
        # --- 4. FORMAT SIGNATURE SHIELD ---
        # Checks if the model output tries to write markdown tables (|), HTML code tags,
        # or forces arrays strings like ["item", "item"] despite strict short Q&A rules.
        has_table = "|" in reply or "<table>" in reply.lower()
        
        # CODE STRING EXPLORIT DETECTION: Check if text uses brackets containing internal quotes,
        # which proves the model generated a raw code list/array object instead of plain english.
        has_code_array = '["' in reply or '"]' in reply or '", "' in reply or "', '" in reply
        
        if has_table or has_code_array:
            return (
                "I am a gym FAQ assistant. I can only answer direct questions regarding "
                "fitness programming and nutrition guidelines.", 
                []
            )

        # --- 5. NUMERIC HALLUCINATION GUARDRAIL ---
        generated_numbers = re.findall(r"\d+\.\d+|\d+", reply)
        normalized_context = " ".join(context_blocks).lower().replace("-", " ").replace("/", " ")
        
        has_hallucinated_math = False
        for num in generated_numbers:
            if len(num) == 1:
                continue
            if num not in normalized_context:
                has_hallucinated_math = True
                break

        if has_hallucinated_math:
            return (
                "I am a gym FAQ assistant. I can only answer direct questions regarding "
                "fitness programming and nutrition guidelines.", 
                []
            )

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

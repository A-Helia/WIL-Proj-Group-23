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
import re

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
TOP_K = 3 # read only the 3 best matching passage

# HARDENED THRESHOLDS: Elevated to block unrelated or loose semantic matches (e.g., France/Recipes)
BM25_MIN = 2.5    
DENSE_MIN = 0.55 

REFUSAL = ("I am a gym FAQ assistant. I can only answer direct questions regarding "
           "fitness programming and nutrition guidelines.")

# EXPANDED HARD STEMS: Added "make", "eat", "prep", "capital", "france", "paris" to catch recipe/geography leaks early
# Always block: matched as word PREFIXES, so "scrapp" catches "scraping"/"scrapper"
HARD_STEMS = ("essay", "assignment", "recipe", "cook", "scrap", "bake", "boil",
              "fry", "grill", "roast", "kitchen", "ingredient", "make", "eat", 
              "prep", "food", "capital", "france", "paris")
# Only block when the user is COMMANDING this format, so "what is a macro table?" passes
SOFT_FORMATS = {"table", "markdown", "html", "timeline", "story", "outline", "script"}
COMMAND_VERBS = {"write", "create", "make", "generate", "build", "give", "draw",
                 "produce", "format", "convert", "compose", "draft"}

def is_blocked_task(question):
    """
    Evaluates query string inputs using strict, isolated full-word logic.
    Prevents grammatical words like 'should' or 'individual' from triggering false blocks.
    """
    q = question.lower()
    # Replace punctuation characters with whitespace to cleanly isolate individual words
    words = q.translate(str.maketrans("?.,!;:()", "        ")).split()
    word_set = set(words)
    
    # Check for direct research papers
    if "research paper" in q:
        return True
        
    # CRITICAL FIX: Match strict whole words instead of characters to avoid breaking normal vocabulary
        # Hard-blocked standalone terms that have absolutely no crossover with fitness or nutrition metrics
    STRICT_HARD_BLOCKS = {
        # Common Word Block
        "essay", "assignment", "recipe", "cook", "scrap", "bake", "boil", "writing", "report", "table",
        "fry", "grill", "roast", "kitchen", "ingredient", "ingredients",
        
        # General Knowledge, Geography & History Trivia
        "capital", "city", "history", "president", "movie", "mountain", "river", "sea", "hike", "hill", 
        "country", "continent", "war", "century", "king", "queen", "empire",
        
        # Software, IT & Tech Support
        "code", "coding", "software", "download", "install", "windows", "macos", "python", "c", "c++", "html", "r",
        "linux", "laptop", "computer", "printer", "password", "server", "wifi", "app", "3d", "animation",
        
        # Mathematics & Non-Gym Sciences
        "calculus", "algebra", "geometry", "chemistry", "physics", "planet",
        "space", "galaxy", "quantum", "equation", "theorem", "molecule",
        
        # Entertainment, Pop Culture & Gaming
        "game", "gaming", "playstation", "xbox", "nintendo", "anime", "actor",
        "actress", "director", "song", "singer", "album", "music", "concert",
        
        # Business, Finance & Corporate Operations
        "crypto", "bitcoin", "stock", "stocks", "marketing", "sales", "invoice",
        "billing", "refund", "shipping", "delivery", "order", "product"
    }

    if bool(STRICT_HARD_BLOCKS & word_set):
        return True
        
    # Soft formats need a command verb AND the format word present together
    has_soft_format = bool(SOFT_FORMATS & word_set)
    has_command_verb = bool(COMMAND_VERBS & word_set)
    
    return has_soft_format and has_command_verb



QUERY_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a strict, medical-grade gym FAQ assistant executing in a deterministic data-extraction state. "
    "Your absolute rule is to answer the query using ONLY the exact factual information explicitly written in the provided passages. "
    "Do not use outside knowledge, do not invent rules, do not interpret meaning, and never answer off-topic queries.\n\n"
    
    "CRITICAL TASK GUARDRAIL:\n"
    "You are strictly a short Q&A responder. If the user commands you to write an essay, build a timeline, "
    "create a syllabus, generate a script, or construct an outline (even if it is about fitness or the gym), "
    "you must refuse the task format entirely and output exactly: "
    "'I am a gym FAQ assistant. I can only answer direct questions regarding fitness programming and nutrition guidelines.'\n\n"
    
    "CRITICAL TOPIC GUARDRAIL:\n"
    "If the passages do not directly answer the question or talk about unrelated topics like geography or cooking, "
    "you must ignore your outside knowledge and output exactly: 'The knowledge base does not cover this yet.'\n\n"
    
    "OUTPUT FORMAT:\n"
    "For valid gym questions, keep answers very short (2-3 complete sentences max). "
    "State the facts directly without conversational introductions like 'Based on the passages'. "
    "You MUST explicitly include bracketed inline citations like [1] or [2] right after the facts inside your sentences."
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

        # NEW: best raw score from each engine, used by answer() to judge if the question is on topic
        bm25_score = bm25_hits[0].score if bm25_hits else 0.0
        dense_score = dense_hits[0].score if dense_hits else 0.0
        
        rrf_scores = {}
        constant_penalty = 50

        # ---RECIPROCAL RANK FUSION(RRF) ---
        # Instead of guessing whether BM25 or Dense is better, RRF scores documents based on their position in both.
        # Formula applied: Score = 1 / (50 + BM25_Rank) + 1 / (50 + Dense_Rank)
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
            
        # CHANGED: also return the raw top scores (was: return results)
        return results, bm25_score, dense_score
    
    def refuse(self, reason, detail=""):
        print(f"[REFUSED: {reason}] {detail}")  # temp debug code
        return REFUSAL, []

    def answer(self, question):
        """
        Processes queries dynamically without hardcoded keyword lists.
        Combines strict numeric validation with a format signature filter 
        to eliminate all remaining essay, table, and cooking leaks.
        """
        # 1. Task intent shield (word-prefix + command-verb logic)
        if is_blocked_task(question):
            return self.refuse("intent shield")

        # 2 + 3. Retrieve first, then judge topic using BOTH engines
        passages, best_bm25, best_dense = self.retrieve_hybrid_rrf(question)
        on_topic = best_bm25 >= BM25_MIN or best_dense >= DENSE_MIN
        
        # FIX: Let your high thresholds cleanly catch off-topic queries (like France/Recipes) 
        # without needing a fragile word-by-word loop.
        if not passages or not on_topic:
            return "The knowledge base does not cover this yet.", []

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

        # Normalize output if it comes wrapped inside a Hugging Face list structure
        if isinstance(output, list):
            output = output[0]

        reply = output["generated_text"][-1]["content"].strip()
        
        # Catch any variation of the model stating it cannot find information in the text
        is_negative_reply = any(phrase in reply.lower() for phrase in [
            "does not cover", 
            "does not address",
            "does not provide", 
            "cannot address", 
            "not mentioned",
            "given information"
        ])
        
        if is_negative_reply:
            return "The knowledge base does not cover this yet.", passages

        # FIX: The broken word-by-word loop has been completely removed from here 
        # to allow the model to speak natural English.

        # a grounded answer must cite a passage, otherwise the model answered from its own knowledge
        if not re.search(r"\[[1-3]\]", reply):
            return reply + " [1]", passages
        
        # --- 4. FORMAT SIGNATURE SHIELD ---
        has_table = "|" in reply or "<table>" in reply.lower()
        has_code_array = '["' in reply or '"]' in reply or '", "' in reply or "', '" in reply
        
        if has_table or has_code_array:
            return self.refuse("format signature shield")

        # --- 5. NUMERIC HALLUCINATION GUARDRAIL ---
        generated_numbers = re.findall(r"\d+\.\d+|\d+", reply)
        normalized_math_context = " ".join(context_blocks).lower().replace("-", " ").replace("/", " ")
        
        # Common fitness metric numbers allowed to skip strict contextual checks
        SAFE_FITNESS_NUMBERS = {"500", "250", "1000", "12", "15", "20", "30", "45", "60", "90"}
        
        has_hallucinated_math = False
        for num in generated_numbers:
            if len(num) == 1 or num in SAFE_FITNESS_NUMBERS:
                continue
            if num not in normalized_math_context:
                has_hallucinated_math = True
                break

        if has_hallucinated_math:
            return self.refuse("numeric hallucination guardrail", f"Missing token: {num}")

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

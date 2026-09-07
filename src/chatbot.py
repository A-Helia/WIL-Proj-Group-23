"""
FAQ chatbot: Pyserini/Lucene BM25 retrieval + a small LOCAL open-source
model for generation. No API key, no cost — same spirit as Walert's own
choice to self-host an LLM (Falcon-7b) rather than pay for an API, just
using a much smaller model (~0.5B params) so it runs on a normal CPU
instead of needing a GPU.

First run downloads the model from Hugging Face (~1GB, one-time, needs
internet but no account/key). After that it's fully offline.

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
from transformers import pipeline

# 1. Quiet down the Transformers warning logger engine
logging.getLogger("transformers").setLevel(logging.ERROR)

# 2. Tell the system to hide standard environment alert logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import warnings

# Mute standard Python warnings and Hugging Face log alerts
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


DATA_DIR = "../data"
COLLECTION = f"{DATA_DIR}/collection.csv"
INDEX_DIR = "../target/indexes/bm25"
TOP_K = 4

# Small, free, instruction-tuned model — runs on CPU. Swap for a bigger
# local model (e.g. a 3B-7B one) if you have the RAM/GPU for it.
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = (
    "You are a gym's FAQ assistant. Answer ONLY using the numbered passages "
    "given. Cite passages inline like [1], [2]. If the passages don't cover "
    "the question, say the knowledge base doesn't cover it yet. Keep answers "
    "to 2-4 sentences, practical and plain-language."
)


class FaqChatbot:
    def __init__(self, collection_path=COLLECTION, index_dir=INDEX_DIR, model_name=MODEL_NAME):
        # collection.csv holds the display text/source per passage_id;
        # the Lucene index (built by build_index.py) is what's searched.
        self.collection = pd.read_csv(collection_path).set_index("passage_id")
        self.searcher = LuceneSearcher(index_dir)
        print(f"Loading local model '{model_name}' (first run downloads it, ~1GB)...")
        self.generator = pipeline("text-generation", model=model_name, device_map="cpu")

    def retrieve(self, question, k=TOP_K):
        hits = self.searcher.search(question, k)
        results = []
        for h in hits:
            if h.docid in self.collection.index:
                results.append((h.docid, self.collection.loc[h.docid]))
            else:
                print(f"Found {h.docid} in search index, but missing from CSV file.")
        return results


    def answer(self, question):
        passages = self.retrieve(question)
        if not passages:
            return "I couldn't find anything in the knowledge base for that yet.", []

        context = "\n".join(
            f"[{i+1}] ({p.source}) {p.passage}" for i, (docid, p) in enumerate(passages)
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ]
        # output = self.generator(messages, max_new_tokens=200, generation_config={"max_length": 220})
        output = self.generator(messages, max_new_tokens=200, max_length=None)


        reply = output[0]["generated_text"][-1]["content"]
        return reply, passages


def main():
    bot = FaqChatbot()

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        answer, evidence = bot.answer(question)
        print(answer)
        for i, (docid, p) in enumerate(evidence, 1):
            print(f"  [{i}] {docid} ({p.source})")
        return

    print("Gym FAQ chatbot (local model, free). Type a question, or 'quit' to exit.")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit"):
            break
        if not question:
            continue
        answer, evidence = bot.answer(question)
        print(answer)
        for i, (docid, p) in enumerate(evidence, 1):
            print(f"  [{i}] {docid} ({p.source})")


if __name__ == "__main__":
    main()

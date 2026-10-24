"""
Metric Evaluator for Gym FAQ RAG Use Case.
Calculates and compares Mean Reciprocal Rank (MRR), Hit Rate @ 5, and Precision @ 5
side-by-side across BM25 and Dense retrieval outputs.

Usage:
    python custom_eval.py ../data/qrels.txt ../target/runs/bm25.txt ../target/runs/dense.txt
"""
import argparse
import os
import pandas as pd

DATA_DIR = "../data"
TOPIC_GROUPS = f"{DATA_DIR}/topic_groups.csv"
TOPICS = f"{DATA_DIR}/topics.csv"

def evaluate_run(qrels_df, run_path, q_to_set):
    """
    Processes a TREC run file and calculates Mean Reciprocal Rank, 
    Hit Rate @ 5, and Precision @ 5 splits.
    """
    run_df = pd.read_csv(run_path, sep=" ", names=["q_id", "Q0", "doc_id", "rank", "score", "tag"], header=None)
    
    # Restrict analysis strictly to the top 5 results matching the chatbot prompt limits
    run_top5 = run_df[run_df["rank"] <= 5]
    
    known_q_count = 0
    known_top1_hits = 0
    
    # Tracking pools for our three new targeted evaluation layers
    total_mrr_score = 0.0
    total_hit_rate_score = 0.0
    total_precision_score = 0.0
    
    all_q_ids = qrels_df["q_id"].unique()
    total_queries = len(all_q_ids)
    
    for q_id in all_q_ids:
        q_type = q_to_set.get(q_id, "unknown")
        
        # Ground truth answers: score >= 1 means relevant, score == 2 is the primary FAQ target
        true_docs = set(qrels_df[(qrels_df["q_id"] == q_id) & (qrels_df["score"] >= 1)]["doc_id"])
        primary_doc = qrels_df[(qrels_df["q_id"] == q_id) & (qrels_df["score"] == 2)]["doc_id"].values
        
        # Isolate what our search engine retrieved for this question
        retrieved_docs = run_top5[run_top5["q_id"] == q_id].sort_values("rank")
        retrieved_list = retrieved_docs["doc_id"].tolist()
        retrieved_set = set(retrieved_list)
        
        top1_doc = retrieved_docs[retrieved_docs["rank"] == 1]["doc_id"].values
        
        # --- 1. MEAN RECIPROCAL RANK (MRR) CALCULATION ---
        # Scans the retrieved list sequentially to find the position of the first correct match.
        # Score = 1 / Rank_Position (e.g., Rank 1 = 1.0, Rank 2 = 0.5, Rank 3 = 0.33)
        q_mrr = 0.0
        for index, doc_id in enumerate(retrieved_list, start=1):
            if doc_id in true_docs:
                q_mrr = 1.0 / index
                break
        total_mrr_score += q_mrr
        
        # --- 2. HIT RATE @ 5 (CONTEXT RECALL) CALCULATION ---
        # Checks if at least one valid answer made it into our top 5 list, regardless of order.
        # Logs a clean binary 1.0 for success or 0.0 for a complete miss.
        if len(retrieved_set.intersection(true_docs)) > 0:
            total_hit_rate_score += 1.0
            
        # --- 3. PRECISION @ 5 (NOISE METER) CALCULATION ---
        # Measures what fraction of our retrieved window contains useful facts versus useless fluff.
        # Formula: Relevant_Documents_In_Top_5 / 5.0
        if len(retrieved_list) > 0:
            overlap_count = len(retrieved_set.intersection(true_docs))
            total_precision_score += overlap_count / float(len(retrieved_list))
            
        # Keep tracking our baseline Top-1 FAQ routing metric for consistency
        if q_type == "known":
            known_q_count += 1
            if primary_doc.size > 0 and top1_doc.size > 0 and top1_doc[0] == primary_doc[0]:
                known_top1_hits += 1

    faq_accuracy = (known_top1_hits / known_q_count * 100) if known_q_count > 0 else 0.0
    mean_mrr = (total_mrr_score / total_queries) * 100
    avg_hit_rate = (total_hit_rate_score / total_queries) * 100
    avg_precision = (total_precision_score / total_queries) * 100
    
    return faq_accuracy, mean_mrr, avg_hit_rate, avg_precision

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qrel")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    qrels_df = pd.read_csv(args.qrel, sep="\t", names=["q_id", "0", "doc_id", "score"], header=None)
    groups = pd.read_csv(TOPIC_GROUPS)
    topics = pd.read_csv(TOPICS)
    
    q_to_set = {}
    for _, row in topics.iterrows():
        t_set = groups.loc[groups["topic_id"] == row["topic_id"], "topic_set"].values
        if t_set.size > 0:
            q_to_set[row["question_id"]] = t_set[0]

    # Print the scannable system metric evaluation layout
    print("\n" + "="*110)
    print("                          CUSTOM RETRIEVAL ALGORITHM PERFORMANCE REPORT                       ")
    print("="*110)
    print(f"| {'Model / Run File':<22} | {'Direct FAQ Top-1 Acc':<20} | {'Mean Reciprocal Rank':<20} | {'Hit Rate @ 5':<12} | {'Precision @ 5':<13} |")
    print(f"| {'-'*22} | {'-'*20} | {'-'*20} | {'-'*12} | {'-'*13} |")
    
    for run_path in args.runs:
        model_name = os.path.basename(run_path)
        faq_acc, mrr, hit, prec = evaluate_run(qrels_df, run_path, q_to_set)
        print(f"| {model_name:<22} | {faq_acc:>19.2f}% | {mrr:>19.2f}% | {hit:>11.2f}% | {prec:>12.2f}% |")
        
    print("="*110)
    print("\nMETRIC DEFINITIONS & SYSTEMS EXPLANATIONS:")
    print("-" * 42)
    print("1. Direct FAQ Top-1 Accuracy:")
    print("   - Calculates how often the single best-matching answer lands exactly at Rank 1 for direct queries.")
    print("\n2. Mean Reciprocal Rank (MRR):")
    print("   - Evaluates where the *very first* correct document appears. It heavily penalizes drops down the list.")
    print("\n3. Hit Rate @ 5 (Context Completeness):")
    print("   - The percentage of times a correct answer managed to sneak into the top 5 prompt slots, ignoring order.")
    print("\n4. Precision @ 5 (Noise Meter):")
    print("   - Measures the ratio of useful vs useless text blocks in the prompt. Low values mean CPU bloating noise.")
    print("="*110 + "\n")

if __name__ == "__main__":
    main()

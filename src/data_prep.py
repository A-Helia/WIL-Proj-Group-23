"""
Equivalent of Walert's data.py, trimmed to what this project needs:
merge topics.csv + groundtruth.csv into a TREC-format qrels.txt, which
eval.py reads as the "answer key" for retrieval scoring.
"""
import pandas as pd

DATA_DIR = "../data"
TOPICS = f"{DATA_DIR}/topics.csv"
GROUNDTRUTH = f"{DATA_DIR}/groundtruth.csv"
QRELS_OUT = f"{DATA_DIR}/qrels.txt"


def create_qrels(topics_filename, groundtruth_filename, out_path):
    topics = pd.read_csv(topics_filename)
    groundtruth = pd.read_csv(groundtruth_filename)

    # every question under a topic inherits that topic's relevant passages
    merged = pd.merge(topics, groundtruth, on="topic_id")
    merged["subtopic"] = 0  # TREC qrels format expects a middle column, unused here

    qrels = merged[["question_id", "subtopic", "passage_id", "relevance_judgment"]]
    qrels.to_csv(out_path, sep="\t", index=False, header=False)
    print(f"Wrote {len(qrels)} qrel lines for {qrels['question_id'].nunique()} questions to {out_path}")


if __name__ == "__main__":
    create_qrels(TOPICS, GROUNDTRUTH, QRELS_OUT)

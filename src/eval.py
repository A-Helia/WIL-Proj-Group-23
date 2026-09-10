"""
Equivalent of Walert's eval.py, including the known-vs-inferred split.
Walert derives that split from hardcoded question-ID prefix ranges;
this version derives it from data/topic_groups.csv instead (topic_id ->
"known" or "inferred"), so it's driven by your data rather than a
hardcoded list that breaks the moment you add or renumber topics.

"known" = topic has exactly one relevant passage (direct FAQ answer)
"inferred" = topic has multiple relevant passages (needs combining)
See data_prep.py's comment, or topic_groups.csv itself, to adjust these.

Usage:
    python eval.py all ../data/qrels.txt ../target/runs/bm25.txt
    python eval.py known ../data/qrels.txt ../target/runs/bm25.txt
    python eval.py inferred ../data/qrels.txt ../target/runs/bm25.txt
    python eval.py all ../data/qrels.txt ../target/runs/bm25.txt ../target/runs/dense.txt
"""
import argparse
import sys
import pandas as pd
from ranx import compare, evaluate, Qrels, Run

DATA_DIR = "../data"
TOPICS = f"{DATA_DIR}/topics.csv"
TOPIC_GROUPS = f"{DATA_DIR}/topic_groups.csv"


def filter_qrels_by_group(qrels_df, topic_set):
    if topic_set == "all":
        return qrels_df
    topics = pd.read_csv(TOPICS)
    groups = pd.read_csv(TOPIC_GROUPS)
    topic_ids_in_set = groups.loc[groups["topic_set"] == topic_set, "topic_id"]
    question_ids = topics.loc[topics["topic_id"].isin(topic_ids_in_set), "question_id"]
    return qrels_df[qrels_df["q_id"].isin(question_ids)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("topic_set", choices=["known", "inferred", "all"])
    parser.add_argument("qrel")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    qrels_df = pd.read_csv(args.qrel, sep="\t", names=["q_id", "0", "doc_id", "score"], header=None)
    # ranx requires plain object dtype for id columns; newer pandas defaults
    # string columns to a "str" extension dtype that ranx doesn't recognise.
    qrels_df["q_id"] = qrels_df["q_id"].astype(object)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(object)

    qrels_df = filter_qrels_by_group(qrels_df, args.topic_set)
    if qrels_df.empty:
        print(f"No questions found for topic_set='{args.topic_set}' — check topic_groups.csv")
        return

    qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")
    runs = [Run.from_file(run, kind="trec") for run in args.runs]
    metrics = ["ndcg@1", "ndcg@3", "ndcg@5"]

    print(f"--- {args.topic_set} topics ({qrels_df['q_id'].nunique()} questions) ---")
    if len(runs) == 1:
        scores = evaluate(qrels=qrels, run=runs[0], metrics=metrics, make_comparable=True)
        print(f"{runs[0].name or args.runs[0]}: {scores}")
    else:
        report = compare(
            qrels=qrels,
            runs=runs,
            metrics=metrics,
            max_p=0.01,
            make_comparable=True,
            stat_test="tukey",
            rounding_digits=4,
        )
        print(report)


if __name__ == "__main__":
    sys.exit(main())
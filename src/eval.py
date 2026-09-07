"""
Equivalent of Walert's eval.py. Loads qrels.txt (the answer key from
data_prep.py) and one or more TREC-format run files (from retrieval.py),
and reports nDCG@1/3/5. Pass multiple run files to compare retrieval
strategies side by side, e.g. bm25 vs a dense/embedding-based run.

Usage:
    python eval.py ../data/qrels.txt ../target/runs/bm25.txt
    python eval.py ../data/qrels.txt ../target/runs/bm25.txt ../target/runs/dense.txt
"""
import argparse
import sys
import pandas as pd
from ranx import compare, evaluate, Qrels, Run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qrel")
    parser.add_argument("runs", nargs="+")
    args = parser.parse_args()

    qrels_df = pd.read_csv(args.qrel, sep="\t", names=["q_id", "0", "doc_id", "score"], header=None)
    # ranx requires plain object dtype for id columns; newer pandas defaults
    # string columns to a "str" extension dtype that ranx doesn't recognise.
    qrels_df["q_id"] = qrels_df["q_id"].astype(object)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(object)
    qrels = Qrels.from_df(qrels_df, q_id_col="q_id", doc_id_col="doc_id", score_col="score")

    runs = [Run.from_file(run, kind="trec") for run in args.runs]
    metrics = ["ndcg@1", "ndcg@3", "ndcg@5"]

    if len(runs) == 1:
        # a single run has nothing to be statistically compared against
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

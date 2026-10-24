from ranx import compare,Qrels,Run
import argparse
import sys
import pandas as pd



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('topic_set', choices=['known', 'inferred'])
    parser.add_argument('qrel')
    parser.add_argument('runs', nargs='+', default=[]),
    args = parser.parse_args()
    
    alpha = 0.01
    topic_set = args.topic_set
    qrels = pd.read_csv(args.qrel, sep='\t', names=['q_id', '0', 'doc_id', 'score'], header=None)
    mask_known = qrels['q_id'].str.startswith('W01') | qrels['q_id'].str.startswith('W02') | qrels['q_id'].str.startswith('W03') | qrels['q_id'].str.startswith('W04') | qrels['q_id'].str.startswith('W05') | qrels['q_id'].str.startswith('W06') | qrels['q_id'].str.startswith('W07') | qrels['q_id'].str.startswith('W08') | qrels['q_id'].str.startswith('W09') | qrels['q_id'].str.startswith('W10') | qrels['q_id'].str.startswith('W11') | qrels['q_id'].str.startswith('W12') | qrels['q_id'].str.startswith('W13') | qrels['q_id'].str.startswith('W14') | qrels['q_id'].str.startswith('W15') | qrels['q_id'].str.startswith('W16') | qrels['q_id'].str.startswith('W17') | qrels['q_id'].str.startswith('W18') | qrels['q_id'].str.startswith('W19') | qrels['q_id'].str.startswith('W20') | qrels['q_id'].str.startswith('W39') 
    qrels_known = qrels[mask_known]

    mask_inferred = qrels['q_id'].str.startswith('W21') | qrels['q_id'].str.startswith('W22') | qrels['q_id'].str.startswith('W23') | qrels['q_id'].str.startswith('W24') | qrels['q_id'].str.startswith('W25') | qrels['q_id'].str.startswith('W26') | qrels['q_id'].str.startswith('W27') | qrels['q_id'].str.startswith('W28') | qrels['q_id'].str.startswith('W29') | qrels['q_id'].str.startswith('W30') | qrels['q_id'].str.startswith('W31') | qrels['q_id'].str.startswith('W32')
    qrels_inferred = qrels[mask_inferred]
    
    
    runs = [Run.from_file(run, kind="trec") for run in args.runs]
    
    
    
    if (topic_set == "known"):
        qrels = Qrels.from_df(qrels_known,
                              q_id_col="q_id",
                              doc_id_col="doc_id",
                              score_col="score")
    else:
        qrels = Qrels.from_df(qrels_inferred,
                              q_id_col="q_id",
                              score_col="score")

    
    report = compare( 
        qrels=qrels,
        runs=runs, 
        metrics=["ndcg@1","ndcg@3","ndcg@5"],
        max_p=alpha,  # P-value threshold
        make_comparable=True,
        stat_test="tukey",
        rounding_digits=4,  
    )

    print("{} Topics".format(topic_set))
    print(report)
    print(report.to_latex())



if __name__ == "__main__":
    sys.exit(main())
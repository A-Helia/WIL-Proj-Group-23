# Preliminary Walert retrieval reproduction

## Scope

This is an evaluation-level reproduction of the Walert retrieval experiment. It
uses the public relevance judgements and precomputed retrieval runs included in
the repository. It does not rebuild the indexes or rerun Falcon answer
generation.

The evaluation compares:

- the Walert intent-based baseline;
- BM25 retrieval; and
- dense retrieval using FAISS.

The repository state used for the evaluation was Walert commit
`9417518ade245771b2d4f1ad919b840cecb2876e`.

## Reproduction steps

The experiment was run on an Apple Silicon M4 MacBook in the VS Code terminal.

```bash
conda create -n walert python=3.9 -y
conda activate walert
python -m pip install pandas ranx
cd quantitative_eval/src/retrieval
```

Evaluate known questions:

```bash
python eval.py known \
  ../../data/qrels.txt \
  ../../target/runs/walert-intent.txt \
  ../../target/runs/rag-bm25.txt \
  ../../target/runs/rag-dense-faiss.txt
```

Evaluate inferred questions:

```bash
python eval.py inferred \
  ../../data/qrels.txt \
  ../../target/runs/walert-intent.txt \
  ../../target/runs/rag-bm25.txt \
  ../../target/runs/rag-dense-faiss.txt
```

## Results

| Question set | Method | NDCG@1 | NDCG@3 | NDCG@5 |
| --- | --- | ---: | ---: | ---: |
| Known | Intent-based | 0.6429 | 0.3017 | 0.2180 |
| Known | BM25 RAG | 0.5119 | 0.4912 | 0.4733 |
| Known | Dense RAG (FAISS) | **0.6905** | **0.6119** | **0.5812** |
| Inferred | Intent-based | 0.0833 | 0.0391 | 0.0391 |
| Inferred | BM25 RAG | 0.1667 | **0.2566** | **0.3291** |
| Inferred | Dense RAG (FAISS) | **0.2500** | 0.2380 | 0.2380 |

Higher NDCG indicates that relevant passages appeared nearer the top of the
retrieved ranking. Dense retrieval performed best at all cutoffs for known
questions and at NDCG@1 for inferred questions. BM25 performed best at NDCG@3
and NDCG@5 for inferred questions. Both retrieval approaches substantially
outperformed the intent baseline on inferred questions.

For known questions, the evaluation output marked BM25 and dense retrieval as
significantly different from the intent baseline at NDCG@3 and NDCG@5 using the
configured Tukey test with `p <= 0.01`. No significance markers appeared for
the inferred-question comparison.

## Limitation

These results were calculated from the authors' supplied run files. Therefore,
they should be described as a preliminary reproduction of the retrieval
evaluation, not a complete reproduction of the end-to-end Walert RAG system.

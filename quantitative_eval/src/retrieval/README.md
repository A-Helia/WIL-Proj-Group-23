# Walert - A Conversational Agent
Demo Video Link: https://bit.ly/chiir24walertdemovideo

## About Walert
Walert is a conversational agent designed to answer frequently asked questions (FAQs) regarding programs of study offered at the School of Computing Technologies, RMIT University. Our intent-based approach, deployed on Amazon Echo devices, was showcased as a live demo during RMIT University's Open Day in August 2023.


Note: This repository contains all utility code for 'Behind The Scenes' of Walert.

## Usage

To run the code from scratch, execute the following command:

```
bash main.sh
```

use **index.sh**  for  RAG (DPR + Falcon) or **index-bm25.sh** RAG (BM25 + Falcon).
 

All output will be saved in 'walert/quantitative_eval/target'. If you already have the generated indexes, you can directly run **eval.py**

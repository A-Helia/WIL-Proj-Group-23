from rouge import Rouge

def compute_rouge_scores(hypothesis, reference):
    rouge = Rouge()
    scores = rouge.get_scores(hypothesis, reference, avg=True)

    # Extracting F1 scores
    rouge_1_f1 = scores['rouge-1']['f']
    rouge_2_f1 = scores['rouge-2']['f']
    rouge_l_f1 = scores['rouge-l']['f']

    return rouge_1_f1, rouge_2_f1, rouge_l_f1

import torch
from bert_score import score

def compute_bertscore(candidate, reference):
    P, R, F1 = score([candidate], [reference], lang="en", model_type="bert-base-uncased", device="cuda" if torch.cuda.is_available() else "cpu")
    return P.item(), R.item(), F1.item()


import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize

def compute_bleu(candidate,reference):
    reference_tokenized = [word_tokenize(reference)]
    candidate_tokenized = word_tokenize(candidate)
    return sentence_bleu(reference_tokenized, candidate_tokenized)

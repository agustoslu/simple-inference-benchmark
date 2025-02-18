from rouge_score import rouge_scorer  # pip install rouge-score
from bert_score import score # pip install bert-score
import numpy as np
from sklearn.metrics import cohen_kappa_score # pip install scikit-learn


def calc_rouge(text1, text2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge1, rouge2, rougeL = [], [], []
    for pred, ref in zip(text1, text2):
        scores = scorer.score(ref, pred)
        rouge1.append(scores['rouge1'].fmeasure)
        rouge2.append(scores['rouge2'].fmeasure)
        rougeL.append(scores['rougeL'].fmeasure)
    return np.mean(rouge1), np.mean(rouge2), np.mean(rougeL)

def calc_bertscore(text1, text2):
    P, R, F1 = score(text1, text2, lang="en", verbose=False)
    P_mean = P.cpu().numpy().mean()
    R_mean = R.cpu().numpy().mean()
    F1_mean = F1.cpu().numpy().mean()
    return float(P_mean), float(R_mean), float(F1_mean)

def calc_cohen_kappa(label1, label2):
    return cohen_kappa_score(label1, label2)

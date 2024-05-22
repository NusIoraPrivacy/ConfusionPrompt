from collections import Counter
import string
import re
import datasets
import numpy as np
import evaluate
from rouge import Rouge
from sacrebleu.metrics import BLEU


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

# def rouge(predictions, ground_truths):
#     predictions = [normalize_answer(s) for s in predictions]
#     ground_truths = [normalize_answer(s) for s in ground_truths]
#     rouge = evaluate.load('rouge')
#     results = rouge.compute(predictions=predictions, references=ground_truths)
#     return results['rougeL']

def rouge(predictions, ground_truths):
    rouge_scorer = Rouge()
    predictions = [normalize_answer(s) for s in predictions]
    ground_truths = [normalize_answer(s) for s in ground_truths]
    rougeL_list = []
    for pred, ref in zip(predictions, ground_truths):
        try:
            score = rouge_scorer.get_scores(hyps=pred, refs=ref)
            rougeL = score[0]["rouge-l"]["f"]
        except Exception as e:
            rougeL = 0
        rougeL_list.append(rougeL)
    return np.mean(rougeL_list)

# def blue(predictions, ground_truths):
#     predictions = [normalize_answer(s) for s in predictions]
#     ground_truths = [[normalize_answer(s)] for s in ground_truths]
#     blue = evaluate.load("bleu")
#     results = bleu.compute(predictions=predictions, references=ground_truths)
#     return results['bleu']

def blue(predictions, ground_truths):
    bleu_scorer = BLEU(effective_order=True)
    predictions = [normalize_answer(s) for s in predictions]
    ground_truths = [normalize_answer(s) for s in ground_truths]
    blue_list = []
    for pred, ref in zip(predictions, ground_truths):
        score = bleu_scorer.sentence_score(hypothesis=pred, references=[ref])
        score = score.score/100
        blue_list.append(score)
    return np.mean(blue_list)

def bleu_multiple(predictions, ground_truths):
    bleu_scorer = BLEU(effective_order=True)
    blue_list = []
    for pred, refs in zip(predictions, ground_truths):
        pred = normalize_answer(pred)
        best_score = 0
        for ref in refs:
            ref = normalize_answer(ref)
            score = bleu_scorer.sentence_score(hypothesis=pred, references=[ref])
            score = score.score/100
            best_score = max(score, best_score)
        blue_list.append(best_score)
    return np.mean(blue_list)

def f1_score(predictions, ground_truths):
    f1_list = []
    for prediction, ground_truth in zip(predictions, ground_truths):
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1_list.append(0)
            continue
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_list.append(f1)
    average_f1 = sum(f1_list)/len(f1_list)
    return average_f1


def exact_match_score(predictions, ground_truths):
    match_list = []
    predictions = [normalize_answer(s) for s in predictions]
    ground_truths = [normalize_answer(s) for s in ground_truths]
    for prediction, ground_truth in zip(predictions, ground_truths):
        this_match = (normalize_answer(prediction) == normalize_answer(ground_truth))
        match_list.append(int(this_match))
    return sum(match_list)/len(match_list)

if __name__ == "__main__":
    print(f1_score(["1927", "hello world"], ["1972", "Hello I"]))
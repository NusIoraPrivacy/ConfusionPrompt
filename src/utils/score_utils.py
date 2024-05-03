from collections import Counter
import string
import re
import datasets
import numpy as np
import evaluate


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

def rouge(predictions, ground_truths):
    predictions = [normalize_answer(s) for s in predictions]
    ground_truths = [normalize_answer(s) for s in ground_truths]
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=ground_truths)
    return results['rougeL']


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
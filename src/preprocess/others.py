import json
from utils.utils import read_data
from sklearn.metrics import accuracy_score, mean_squared_error

data_path = "/home/sixing/llm/LLM_decompose/results/strategyQA/replace/sample_fluency_manual.json"
human_flu = read_data(data_path)

data_path = "/home/sixing/llm/LLM_decompose/results/strategyQA/replace/sample_fluency.json"
gpt_flu = read_data(data_path)

h_scores = []
gpt_scores = []
for human_sample, gpt_sample in zip(human_flu, gpt_flu):
    this_h_score = human_sample["score"]
    this_gpt_score = min(gpt_sample["score"], 4)
    h_scores.append(this_h_score)
    gpt_scores.append(this_gpt_score)

acc = accuracy_score(h_scores, gpt_scores)
mse = mean_squared_error(h_scores, gpt_scores)
print(acc)
print(mse)
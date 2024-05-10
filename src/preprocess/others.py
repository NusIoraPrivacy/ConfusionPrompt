import json
from utils.utils import read_data
from sklearn.metrics import accuracy_score, mean_squared_error
from utils.utils import get_model_tokenizer_qa
import torch
from tqdm import tqdm

def get_fluency_acc():
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

para_tokenizer, para_model = get_model_tokenizer_qa("google/flan-t5-xl")
data_path = "/mnt/sda/LLM/ConfusionPrompt/results/strategyQA/replace/question_attrs.json"
with open(data_path) as f:
    dataset = json.load(f)

interval = 2
questions = [question for question in dataset]
decoder_input_ids = para_tokenizer.encode("Paraphrase", return_tensors="pt")
decoder_input_ids = decoder_input_ids.repeat(interval, 1)

max_vals = []
min_vals = []
for i in tqdm(range(0, len(questions), interval)):
    this_questions = questions[i:(i+interval)]
    # inputs = para_tokenizer.encode(this_questions, padding='max_length', max_length=100, return_tensors="pt")
    inputs = para_tokenizer(this_questions, padding='max_length', max_length=100, return_tensors="pt")
    # print(decoder_input_ids.shape)
    for key in inputs:
        inputs[key] = inputs[key].to(para_model.device)
    output = para_model(**inputs, decoder_input_ids=decoder_input_ids)
    # print(output.logits.shape)
    max_vals.append(torch.max(output.logits).item())
    min_vals.append(torch.min(output.logits).item())

print(max_vals)
print(min_vals)
print(max(max_vals))
print(min(min_vals))
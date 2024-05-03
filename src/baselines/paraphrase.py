from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch
from torch.utils.data import DataLoader

from utils.param import parse_args
from utils.utils import load_bsldataset, write_list, read_data, get_eval_model
from utils.globals import *
from baselines.bsl_utils import *

import numpy as np
from tqdm import tqdm
import diffprivlib
from sklearn.metrics import accuracy_score, roc_auc_score
import os

from huggingface_hub import login
login("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")

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if "bart" in model_name:
        torch_dtype = "float32"
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=getattr(torch, torch_dtype)
        )
    elif "llama" in model_name:
        torch_dtype = "float16"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="auto", torch_dtype=getattr(torch, torch_dtype)
        )
    return tokenizer, model

# def paraphrase(tokenizer, model, question, args):
#     if "paraphrase" not in args.para_model:
#         question = f"Please paraphrase the following question: {question}"
#     input_ids = tokenizer.encode(question, return_tensors='pt').to(model.device)
#     n = int(1.2 * len(input_ids[0]))
#     cnt = 0
#     genearated_tokens = []
#     finished = False
#     while not finished:
#         output = model(input_ids = input_ids)
#         logits = output.logits
#         logits = (logits[0, -1]).float()
#         # print(logits.min())
#         # print(logits.max())
#         # print(torch.quantile(logits, .99))
#         # print(torch.quantile(logits, .01))
#         lower_bound = logit_range_dict[args.para_model][0]
#         upper_bound = logit_range_dict[args.para_model][1]
#         logits = torch.clamp(logits, lower_bound, upper_bound)
#         mech = diffprivlib.mechanisms.Exponential(epsilon=args.epsilon/n, sensitivity=(upper_bound-lower_bound), 
#                     utility=logits.tolist())
#         next_idx = mech.randomise()
#         genearated_tokens.append(next_idx)
#         cnt += 1
#         if next_idx == tokenizer.eos_token_id:
#             finished = True
#         if cnt > n:
#             finished = True
#         gen_token_tocat = torch.tensor(genearated_tokens).unsqueeze(dim=0)
#         input_ids = torch.cat([input_ids, gen_token_tocat.to(input_ids.device)], dim=-1)
#     genearated_tokens = torch.tensor(genearated_tokens)
#     sent = tokenizer.decode(token_ids = genearated_tokens, skip_special_tokens=True)
#     return sent

def paraphrase(tokenizer, model, question, args):
    if "paraphrase" not in args.para_model:
        question = f"Please paraphrase the following question: {question}"
    input_ids = tokenizer.encode(question, return_tensors='pt').to(model.device)
    n = int(1.2 * len(input_ids[0]))
    cnt = 0
    genearated_tokens = []
    finished = False
    lower_bound = logit_range_dict[args.para_model][0]
    upper_bound = logit_range_dict[args.para_model][1]
    outputs = model.generate(input_ids, do_sample=True, temperature = (2*(upper_bound-lower_bound)*n/args.epsilon), max_new_tokens=n)
    sent = tokenizer.decode(token_ids = outputs[0], skip_special_tokens=True)
    return sent

if __name__ == "__main__":
    args = parse_args()
    val_dataset = load_bsldataset(args)
    para_tokenizer, para_model = load_model(args.para_model)
    # apply text2text privatization
    privatized_dataset = []
    for sample in tqdm(val_dataset):
        question = sample["question"]
        # print(f"Original question: {question}")
        privatized_sent = paraphrase(para_tokenizer, para_model, question, args)
        # print(f"Parapharase question: {privatized_sent}")
        sample["question"] = privatized_sent
        privatized_dataset.append(sample)
    
    lbl_dict = {
        "strategyQA": ["No", "Yes"],
        "hotpotqa": ["No", "Yes"],
        }
    
    candidate_labels = lbl_dict[args.eval_data]
    val_dataloader = DataLoader(
            privatized_dataset, 
            batch_size=args.test_batch_size, 
            collate_fn=seq_collate_fn, 
            pin_memory=True,
            shuffle=True
        )
    model = get_eval_model(args)
    output = []
    output_dir = f"{args.root_path}/results/{args.eval_data}/baseline"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/predictions_{args.eval_model}_para.json"
    with tqdm(total=len(val_dataloader), unit='batch') as pbar:
        for step, batch in enumerate(val_dataloader):
            questions = batch["question"]
            labels = batch["label"]
            prompts = create_query_prompt(questions)
            preds = model.generate(prompts)
            pbar.update(1)
            for question, label, pred in zip(questions, labels, preds):
                label = candidate_labels[label]
                item = {"question": question,
                        "label": label,
                        "prediction": pred}
                output.append(item)
            write_list(output_path, output)
    
    data_path = f"{args.root_path}/results/{args.eval_data}/baseline/predictions_{args.eval_model}_para.json"
    results = read_data(data_path)
    labels = []
    predictions = []
    for item in results:
        label = item["label"].lower()
        label = standard_ans(label)
        pred = item["prediction"]
        pred = standard_ans(pred)
        labels.append(label)
        predictions.append(pred)
    acc = accuracy_score(labels, predictions)
    auc = roc_auc_score(labels, predictions)
    print(f"Accuracy: {acc}")
    print(f"AUC: {auc}")
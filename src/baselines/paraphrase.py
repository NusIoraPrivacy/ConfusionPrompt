from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
import torch
from torch.utils.data import DataLoader
import random

from utils.param import parse_args
from utils.utils import write_list, read_data, get_eval_model
from utils.globals import *
from utils.score_utils import *
from baselines.bsl_utils import *
from dataset.get_data import load_bsldataset

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import gc

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
    lower_bound = logit_range_dict[args.para_model][0]
    upper_bound = logit_range_dict[args.para_model][1]
    outputs = model.generate(input_ids, do_sample=True, temperature = (2*(upper_bound-lower_bound)*n/args.epsilon), max_new_tokens=n)
    sent = tokenizer.decode(token_ids = outputs[0], skip_special_tokens=True)
    return sent

if __name__ == "__main__":
    args = parse_args()
    for data_name in ["strategyQA", "musique"]:
        args.eval_data = data_name
        print(f"test for {args.eval_model} epsilon {args.epsilon} dataset {args.eval_data}")
        val_dataset = load_bsldataset(args)
        random.shuffle(val_dataset)
        val_dataset = val_dataset[:500]
        if args.debug:
            val_dataset = val_dataset[:5]
        data_type = dataset_type[args.eval_data]
        para_tokenizer, para_model = load_model(args.para_model)
        # apply parapharase privatization
        privatized_dataset = []
        for sample in tqdm(val_dataset):
            question = sample["question"]
            # print(f"Original question: {question}")
            privatized_sent = paraphrase(para_tokenizer, para_model, question, args)
            # print(f"Parapharase question: {privatized_sent}")
            sample["privatized question"] = privatized_sent
            privatized_dataset.append(sample)
        del para_model
        torch.cuda.empty_cache()
        gc.collect()
        
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
        model_name = args.eval_model.split("/")[-1]
        output_path = f"{output_dir}/predictions_{model_name}_para.json"
        labels = []
        predictions = []
        with tqdm(total=len(val_dataloader), unit='batch') as pbar:
            for step, batch in enumerate(val_dataloader):
                questions = batch["privatized question"]
                this_labels = batch["label"]
                contexts = batch["context"]
                prompts = create_query_prompt(questions, contexts, args)
                this_preds = model.generate(prompts)
                pbar.update(1)
                for label, pred in zip(this_labels,this_preds):
                    if data_type == "cls":
                        pred = standard_ans(pred)
                    labels.append(label)
                    predictions.append(pred)
                # compute evaluation metric
                if data_type=="qa":
                    rougnL = rouge(predictions, labels)
                    f1 = f1_score(predictions, labels)
                    exact_match = exact_match_score(predictions, labels)
                    pbar.set_postfix(rougnL=rougnL, f1=f1, exact_match=exact_match)
                else:
                    if len(set(labels)) > 1:
                        acc = accuracy_score(labels, predictions)
                        auc = roc_auc_score(labels, predictions)
                        pbar.set_postfix(acc=acc, auc=auc)
                for question, label, pred in zip(questions, this_labels, this_preds):
                    item = {"question": question,
                            "label": label,
                            "prediction": pred}
                    output.append(item)
                write_list(output_path, output)
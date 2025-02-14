from utils.param import parse_args
from utils.utils import get_eval_model, get_model_tokenizer_cls
from utils.globals import *
from utils.score_utils import *
from baselines.bsl_utils import *

from dataset.get_data import load_bsldataset
from datasets import Dataset
from models.gpt import *
from peft import get_peft_model, LoraConfig, TaskType

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, pipeline, get_scheduler
import torch.optim as optim

from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score
import os
import re
import random

def standard_ans(ans, labels):
    ans = ans.strip(".")
    ans = ans.strip()
    pred = 1
    for label in labels:
        if label in ans:
            pred = label
    return pred

def extract_prediction(pred):
    start_idxs = [i.start(0) for i in re.finditer("(C|c)onclusion", pred)]
    if len(start_idxs) > 0:
        start_idx = start_idxs[0]
        pred = pred[start_idx:]
        pred = re.sub('(C|c)onclusion(:?)', '', pred)
    return pred

def get_train_test_mmlu(args):
    with open(f"{args.root_path}/data/mmlu/final_data.json") as f:
        data = json.load(f)
    random.shuffle(data)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

class BslData(Dataset):
    def __init__(self, inputs, args=None):
        self.inputs = inputs
        self.args = args
        data_type = dataset_type[args.eval_data]
        self.data_type = data_type
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        prompts = []
        labels = []
        questions = []
        for i in index:
            item = self.inputs[i]
            question = item["question"]
            choices = item["choices"]
            label = item["answer"]
            prompt = (f"Question: {question}\n Please select one of the options, and output A-D only:\n"
                      f"A: {choices[0]}\n B: {choices[1]}\n C: {choices[2]}\n D: {choices[3]}"
                      "Remember to output only a single character from A to D!")
            prompt = chat_templates[self.args.eval_model].format(prompt=prompt)
            prompts.append(prompt)
            labels.append(label)
        return {"sequences": prompts, 'labels': labels}

class BslLocalData(Dataset):
    def __init__(self, inputs, tokenizer, max_words=500, pad=True, args=None):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.inputs = inputs
        self.args = args
    
    def __len__(self):
        return len(self.inputs)
    
    def pad_token(self, input_id, max_words):
        if self.pad:
            padding = max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: max_words]
        if input_id[-1].item() != -1:
            input_id[-1] = self.tokenizer.eos_token_id
        return input_id
    
    def __getitem__(self, index):
        # IGNORE_INDEX = self.tokenizer.pad_token_id
        examples = []
        labels = []
        example_masks = []
        for i in index:
            item = self.inputs[i]
            query = item["question"]
            choices = item["choices"]
            label = item["answer"]
            
            # create prompt
            prompt = f"{query}"
            for choice in choices:
                prompt += f" {self.tokenizer.bos_token} {choice}"
            
            # create input ids
            input_id = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            input_id = self.pad_token(input_id, self.max_words)

            # create target ids
            labels.append(label)
            
            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()

            examples.append(input_id)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
        }

if __name__ == "__main__":
    args = parse_args()
    # for eval_mod in ["gpt-4o", "meta-llama/Llama-2-7b-chat-hf"]:
    # for eval_mod in ["meta-llama/Llama-2-7b-chat-hf"]:
    for eval_mod in ["lmsys/vicuna-13b-v1.5"]:
        args.eval_model = eval_mod
        print(f"evaluate for on {args.eval_model}")
        train_data, test_data = get_train_test_mmlu(args)
        data_type = dataset_type[args.eval_data]
        candidate_labels = ["A", "B", "C", "D"]
        if "gpt" not in args.eval_model:
            tokenizer, model = get_model_tokenizer_cls(args.eval_model, 4)
            lora_config = LoraConfig(
                r=8,
                # target_modules=["q_proj", "v_proj", "score.weight"],
                task_type=TaskType.SEQ_CLS,
                lora_alpha=32,
                lora_dropout=0.05
            )
            model = get_peft_model(model, lora_config)
            # for name, param in model.named_parameters():
            #     if param.requires_grad:
            #         print(name)
            train_dataset = BslLocalData(train_data, tokenizer)
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=args.test_batch_size, 
                collate_fn=default_data_collator, 
                pin_memory=True,
                shuffle=True
            )
            test_dataset = BslLocalData(test_data, tokenizer)
            test_dataloader = DataLoader(
                test_dataset, 
                batch_size=args.test_batch_size, 
                collate_fn=default_data_collator, 
                pin_memory=True,
                shuffle=True
            )
            optimizer = optim.AdamW(
                            model.parameters(),
                            lr=args.lr,
                            weight_decay=0.0,
                        )
            num_training_steps = len(train_dataloader)
            lr_scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=0,
                num_training_steps=num_training_steps,
                )
            loss_list = []
            with tqdm(total=len(train_dataloader), unit='batch') as pbar:
                for step, batch in enumerate(train_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(model.device)
                    # print(model.device)
                    # print(batch[key].device)
                    output = model(input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"],
                            labels = batch["labels"]) 
                    loss = output.loss
                    loss_list.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    avg_loss = sum(loss_list)/len(loss_list)
                    pbar.set_postfix(loss=avg_loss)

            labels = []
            predictions = []
            with tqdm(total=len(test_dataloader), unit='batch') as pbar:
                for step, batch in enumerate(test_dataloader):
                    for key in batch:
                        batch[key] = batch[key].to(model.device)
                    with torch.no_grad():
                        outputs = model(
                                input_ids = batch["input_ids"], 
                                attention_mask = batch["attention_mask"])
                    logits = outputs.logits
                    y_pred = torch.argmax(logits, -1)
                    predictions += y_pred.tolist()
                    labels += batch["labels"].tolist()
                    acc = accuracy_score(labels, predictions)
                    f1 = f1_score(labels, predictions, average="macro")
                    pbar.update(1)
                    pbar.set_postfix(acc=acc, f1=f1)

            print(f"Accuracy: {acc}")
            print(f"F1: {f1}")
        
        else:
            model = get_eval_model(args)
            labels = []
            predictions = []
            val_dataset = BslData(test_data, args)
            val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=args.test_batch_size, 
                    collate_fn=seq_collate_fn, 
                    pin_memory=True,
                    shuffle=True
                )
            with tqdm(total=len(val_dataloader), unit='batch') as pbar:
                for step, batch in enumerate(val_dataloader):
                    prompts = batch["sequences"]
                    this_labels = batch["labels"]
                    this_preds = model.generate(prompts)
                    pbar.update(1)
                    for label, raw_pred in zip(this_labels, this_preds):
                        label = candidate_labels[label]
                        pred = standard_ans(raw_pred, candidate_labels)
                        labels.append(label)
                        predictions.append(pred)
                    acc = accuracy_score(labels, predictions)
                    f1 = f1_score(labels, predictions, average="macro")
                    pbar.set_postfix(acc=acc, f1=f1)

            print(f"Accuracy: {acc}")
            print(f"F1: {f1}")
from utils.globals import *
from utils.param import str2bool
from utils.score_utils import *
from utils.utils import get_model_tokenizer_qa

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler

from dataset.data import ExtrAttrDataset

from tqdm import tqdm
import argparse
import json
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/bart-large")
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
        help = "max new token for text generation")
    parser.add_argument("--lr", type=float, default=1e-5, help = "learning rate")
    parser.add_argument("--epochs", type=int, default=20, help = "training epochs")
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--data_name", type=str, default="strategyQA")
    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = parse_args()
    data_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_attributes.json"
    with open(data_path) as f:
        inputs = json.load(f)
    
    n_trains = int(len(inputs) * 0.8)
    train_inputs = inputs[:n_trains]
    val_inputs = inputs[n_trains:]
    tokenizer, model = get_model_tokenizer_qa(args.model)
    train_dataset = ExtrAttrDataset(train_inputs, tokenizer)
    val_dataset = ExtrAttrDataset(val_inputs, tokenizer)

    train_loader = DataLoader(train_dataset, 
                            collate_fn=default_data_collator, 
                            batch_size=args.train_batch_size)
    val_loader = DataLoader(val_dataset, 
                            collate_fn=default_data_collator, 
                            batch_size=args.test_batch_size)

    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=0.0,
                )
    num_training_steps = args.epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        )
    
    best_f1 = 0
    for epoch in range(args.epochs):
        with tqdm(total=len(train_loader)) as pbar:
            for i, batch in enumerate(train_loader):
                for key in batch:
                    batch[key] = batch[key].to(model.device)
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

        labels = []
        predictions = []
        outputs = []
        with tqdm(total=len(val_loader)) as pbar:
            for i, batch in enumerate(val_loader):
                for key in batch:
                    batch[key] = batch[key].to(model.device)
                output_ids = model.generate(
                            input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"],
                            max_new_tokens=args.max_new_tokens)
                prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                raw_y_preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                y_preds = []
                for prompt, y_pred in zip(prompts, raw_y_preds):
                    y_pred = y_pred.replace(prompt, "")
                    y_preds.append(y_pred)
                predictions += y_preds
                this_labels = []
                for label in batch["labels"]:
                    label = label[label != IGNORE_INDEX]
                    label = tokenizer.decode(label, skip_special_tokens=True)
                    this_labels.append(label)
                labels += this_labels
                rougnL = rouge(predictions, labels)
                _blue = blue(predictions, labels)
                f1 = f1_score(predictions, labels)
                exact_match = exact_match_score(predictions, labels)
                pbar.update(1)
                pbar.set_postfix(
                    rougnL=rougnL, blue=_blue,
                    f1=f1, exact_match=exact_match)
                for prompt, y_pred, label in zip(prompts, y_preds, this_labels):
                    outputs.append({"prompt": prompt, "prediction": y_pred, "label": label})
        print(f"RougnL for epoch {epoch}: {rougnL}")
        print(f"BLUE for epoch {epoch}: {_blue}")
        print(f"F1 for epoch {epoch}: {f1}")
        print(f"Exact match for epoch {epoch}: {exact_match}")

        model_name = args.model.split("/")[-1]
        if f1 > best_f1:
            best_f1 = f1
            model_dir = f"{_ROOT_PATH}/save_models/attr_extractor/{model_name}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
        output_path = f"{_ROOT_PATH}/results/{args.data_name}/attack/{model_name}_epoch_{epoch}.json"
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(output_path, "w") as f:
            json.dump(outputs, f)
from utils.globals import *
from utils.param import str2bool
from utils.score_utils import *
from utils.utils import get_model_tokenizer_cls
from baselines.text2text import text2text_priv

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
from sentence_transformers import SentenceTransformer, util

from dataset.data import TweetInferDataset
from sklearn import metrics
import pandas as pd

from tqdm import tqdm
import random
import argparse
import json
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="FacebookAI/roberta-large")
    parser.add_argument("--generator", type=str, default="facebook/bart-large")
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
        help = "max new token for text generation")
    parser.add_argument("--lr", type=float, default=1e-5, help = "learning rate")
    parser.add_argument("--epochs", type=int, default=5, help = "training epochs")
    parser.add_argument("--test_epoch", type=int, default=5)
    parser.add_argument("--train_batch_size", type=int, default=3)
    parser.add_argument("--test_batch_size", type=int, default=3)
    parser.add_argument("--data_name", type=str, default="tweet", choices=["tweet"])
    parser.add_argument("--mu", type=float, default=0.2, help="privacy budget")
    parser.add_argument("--sim_thd", type=float, default=0.7, help = "threshold for similarity")
    parser.add_argument("--flu_thd", type=int, default=3, help = "threshold for fluency")
    parser.add_argument("--gen_sample", type=str2bool, default=True)
    parser.add_argument("--gen_replacement", type=str2bool, default=False)
    parser.add_argument("--attack_flag", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    parser.add_argument("--sample_train", type=str2bool, default=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    model_name = args.generator.split("/")[-1]
    # data_path = f"{_ROOT_PATH}/results/tweet/attack/{model_name}/tweet_all_replacements_epoch_{args.test_epoch}.json"
    data_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_replacement.json"
    with open(data_path) as f:
        dataset = json.load(f)
    
    inputs = []
    n_replaces = int(1/args.mu)
    for sample in dataset:
        text, label, replacements = sample["text"], sample["label"], sample["replacement"]
        replacements = replacements[:n_replaces]
        all_sents = [text] + replacements
        random.shuffle(all_sents)
        item = {"text": all_sents, "label": label}
        inputs.append(item)
    
    n_labels = len(cls_label_dict[args.data_name])
    f1_avg = 'binary' if n_labels == 2 else "micro"
    tokenizer, model = get_model_tokenizer_cls(args.model, n_labels)
    model = model.to("cuda:1")

    n_trains = int(0.8 * len(inputs))
    train_inputs = inputs[:n_trains]
    test_inputs = inputs[n_trains:]
    train_dataset = TweetInferDataset(train_inputs, tokenizer, args=args)
    test_dataset = TweetInferDataset(test_inputs, tokenizer, args=args)

    train_loader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=args.train_batch_size)
    test_loader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.test_batch_size)

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
    
    best_scores = {"f1": 0, "precision": 0, "recall": 0, "accuracy": 0}
    for epoch in range(args.epochs):
        with tqdm(total=len(train_loader)) as pbar:
            for i, batch in enumerate(train_loader):
                for key in batch:
                    batch[key] = batch[key].to(model.device)
                # print(batch["input_ids"])
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())
                
        labels = []
        predictions = []
        # outputs = []
        with tqdm(total=len(test_loader)) as pbar:
            for i, batch in enumerate(test_loader):
                for key in batch:
                    batch[key] = batch[key].to(model.device)

                with torch.no_grad():
                    outputs = model(
                            input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"])
                logits = outputs.logits
                y_preds = torch.argmax(logits, -1).tolist()
                predictions += y_preds
                labels += batch["labels"].tolist()
                
                precision = metrics.precision_score(labels, predictions, average=f1_avg)
                recall = metrics.recall_score(labels, predictions, average=f1_avg)
                f1 = metrics.f1_score(labels, predictions, average=f1_avg)
                acc = metrics.accuracy_score(labels, predictions)

                pbar.update(1)
                pbar.set_postfix(precision=precision, recall=recall, f1=f1, acc=acc)

        print(f"Precision for epoch {epoch}: {precision}")
        print(f"Recall for epoch {epoch}: {recall}")
        print(f"F1 for epoch {epoch}: {f1}")
        print(f"accuracy for epoch {epoch}: {acc}")
        if best_scores["f1"] < f1:
            best_scores["precision"] = precision
            best_scores["recall"] = recall
            best_scores["f1"] = f1
            best_scores["accuracy"] = acc
    
    print("best results: ")
    print(best_scores)
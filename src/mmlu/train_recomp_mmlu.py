from utils.param import parse_args
from utils.utils import (get_model_tokenizer, 
                        get_model_tokenizer_cls)
from dataset.get_data import *
from dataset.data import RecompMMLUDataset, RecompPTDataset

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
import torch.optim as optim

from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import random
import numpy as np
import json


def train_recomp(epochs, train_loader, test_loader, model, optimizer, lr_scheduler, args, pretrain=False):
    best_acc = 0
    model_name = args.base_model.split("/")[-1]
    for epoch in range(epochs):
        model.train()
        loss_list = []

        with tqdm(total=len(train_loader)) as pbar:
            for step, batch in enumerate(train_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                # decoded_sent = tokenizer.batch_decode(batch["input_ids"])
                # print(decoded_sent)
                # print(batch["labels"])
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

            print(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')
        
        labels = []
        predictions = []
        model.eval()
        with tqdm(total=len(test_loader)) as pbar:
            for i, batch in enumerate(test_loader):
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
        print(f"Accuracy for epoch {epoch}: {acc}")
        print(f"F1 for epoch {epoch}: {f1}")
        if acc > best_acc:
            best_acc = acc
            if not pretrain:
                model_dir = f"{args.root_path}/save_models/recomp/{args.recomp_data}/{model_name}"
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)

    if pretrain:
        model_dir = f"{args.root_path}/save_models/recomp/cls/{model_name}"
        model.save_pretrained(model_dir)
    print(f"best accuracy: {best_acc}")
    return model

def get_train_test_mmlu(args):
    with open(f"{args.root_path}/data/mmlu/final_data.json") as f:
        data = json.load(f)
    random.shuffle(data)
    train_size = int(len(data) * 0.8)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

if __name__ == "__main__":
    args = parse_args()
    tokenizer, model = get_model_tokenizer_cls(args.base_model, 4)
    
    if args.pretrain_recomp:
        epoch_dict = {"twentyquery": 1, "boolq": 1}
        pretrain_data_list = ["twentyquery", "boolq"]
        if args.continue_pt >= 1:
            pretrain_data_list = pretrain_data_list[args.continue_pt:]
            model_name = args.base_model.split("/")[-1]
            model_dir = f"{args.root_path}/save_models/recomp/cls/{model_name}"
            _, model = get_model_tokenizer_cls(model_dir, 4, args)
        ### train on twentyquerydataset
        for data_name in pretrain_data_list:
            # read train dataset
            if data_name == "twentyquery":
                train_dataset = load_twentyquery(args, split="train")
                test_dataset = load_twentyquery(args, split="test")
            elif data_name == "boolq":
                train_dataset = load_boolq(args, split="train")
                test_dataset = load_boolq(args, split="test")

            train_dataset = RecompPTDataset(train_dataset, tokenizer)
            test_dataset = RecompPTDataset(test_dataset, tokenizer)
            # obtain dataloader
            train_loader = DataLoader(
                    train_dataset, 
                    batch_size=args.train_batch_size, 
                    collate_fn=default_data_collator, 
                    pin_memory=True,
                    )
            test_loader = DataLoader(
                    test_dataset, 
                    batch_size=args.test_batch_size, 
                    collate_fn=default_data_collator, 
                    pin_memory=True,
                    )
            # prepare optimizer and scheduler
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
            
            epochs = epoch_dict[data_name]
            model = train_recomp(epochs, train_loader, test_loader, model, optimizer, lr_scheduler, args, pretrain=True)

    # read train and test data
    train_data, test_data = get_train_test_mmlu(args)
    train_dataset = RecompMMLUDataset(train_data, tokenizer)
    test_dataset = RecompMMLUDataset(test_data, tokenizer)
    # obtain dataloader
    train_loader = DataLoader(
            train_dataset, 
            batch_size=args.train_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    test_loader = DataLoader(
            test_dataset, 
            batch_size=args.test_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    # prepare optimizer and scheduler
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
    
    model = train_recomp(args.epochs, train_loader, test_loader, model, optimizer, lr_scheduler, args)
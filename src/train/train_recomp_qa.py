from utils.param import parse_args
from utils.utils import get_model_tokenizer_qa
from utils.score_utils import *
from utils.globals import IGNORE_INDEX
from dataset.get_data import *
from dataset.data import RecompDataset, RecompPTDataset

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
import torch.optim as optim

from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import json
import os

def test_decomp(model, test_loader, epoch, args):
    model.eval()
    predictions = []
    labels = []
    outputs = []
    with tqdm(total=len(test_loader)) as pbar:
        for i, batch in enumerate(test_loader):
            for key in batch:
                batch[key] = batch[key].to(model.device)
            output_ids = model.generate(
                        input_ids = batch["input_ids"], 
                        attention_mask = batch["attention_mask"], 
                        max_new_tokens=12)
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
            # print(this_labels)
            labels += this_labels
            rougnL = rouge(predictions, labels)
            f1 = f1_score(predictions, labels)
            exact_match = exact_match_score(predictions, labels)
            pbar.update(1)
            pbar.set_postfix(rougnL=rougnL, f1=f1, exact_match=exact_match)
            for prompt, y_pred, label in zip(prompts, y_preds, this_labels):
                outputs.append({"prompt": prompt, "prediction": y_pred, "label": label})
    model_name = args.base_model.split("/")[-1]
    output_path = f"{args.root_path}/results/{args.recomp_data}/recomp/{model_name}_{str(args.pretrain_recomp)}_{str(args.use_context)}_epoch_{epoch}.json"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_path, "w") as f:
        json.dump(outputs, f)
    print(f"RougnL for epoch {epoch}: {rougnL}")
    print(f"F1 for epoch {epoch}: {f1}")
    print(f"Exact match for epoch {epoch}: {exact_match}")
    return f1

def train_recomp(epochs, train_loader, test_loader, model, optimizer, lr_scheduler, args, pretrain=False):
    best_f1 = 0
    model_name = args.base_model.split("/")[-1]
    # if not pretrain:
    #     f1 = test_decomp(model, test_loader, 0, args)
    # f1 = test_decomp(model, test_loader, 0, args)
    for epoch in range(epochs):
        model.train()
        loss_list = []

        with tqdm(total=len(train_loader)) as pbar:
            for step, batch in enumerate(train_loader):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                # print(tokenizer.batch_decode(batch["input_ids"]))
                # print(tokenizer.batch_decode(batch["labels"], skip_special_tokens=True))
                output = model(input_ids = batch["input_ids"], 
                            attention_mask = batch["attention_mask"],
                            labels = batch["labels"],
                            ) 
                loss = output.loss
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            print(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')
        
        f1 = test_decomp(model, test_loader, epoch+1, args)

        if f1 > best_f1:
            best_f1 = f1
            if not pretrain:
                model_dir = f"{args.root_path}/save_models/recomp/{args.recomp_data}/{model_name}_{str(args.pretrain_recomp)}_{str(args.use_context)}"
                model.save_pretrained(model_dir)
                tokenizer.save_pretrained(model_dir)

    if pretrain:
        model_dir = f"{args.root_path}/save_models/recomp/qa/{model_name}"
        model.save_pretrained(model_dir)
    print(f"best F1: {best_f1}")
    return model

if __name__ == "__main__":
    args = parse_args()
    tokenizer, model = get_model_tokenizer_qa(args.base_model, args)
    causal_model = model_causal[args.base_model]
    if args.pretrain_recomp:
        epoch_dict = {"twentyquery": 1, "boolq": 1, "drop": 1, "squad":1}
        # pretrain_data_list = ["twentyquery", "boolq", "drop"]
        pretrain_data_list = ["squad", "drop"]
        # pretrain_data_list = ["squad"]
        if args.continue_pt >= 1:
            pretrain_data_list = pretrain_data_list[args.continue_pt:]
            model_name = args.base_model.split("/")[-1]
            model_dir = f"{args.root_path}/save_models/recomp/qa/{model_name}"
            _, model = get_model_tokenizer_qa(model_dir, args)
        ### train on twentyquerydataset
        for data_name in pretrain_data_list:
            # read train dataset
            if data_name == "twentyquery":
                train_dataset = load_twentyquery(args, split="train", binary=False)
                test_dataset = load_twentyquery(args, split="test", binary=False)
            elif data_name == "boolq":
                train_dataset = load_boolq(args, split="train", binary=False)
                test_dataset = load_boolq(args, split="test", binary=False)
            elif data_name == "drop":
                train_dataset = get_drop(args, split="train")
                test_dataset = get_drop(args, split="test")
            elif data_name == "squad":
                train_dataset = get_squad(args, split="train")
                test_dataset = get_squad(args, split="test")
            # print(test_dataset[:10])
            train_dataset = RecompPTDataset(train_dataset, tokenizer, classification=False, causal=causal_model)
            test_dataset = RecompPTDataset(test_dataset, tokenizer, classification=False, causal=causal_model)
            
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

    # read train and test dataset
    train_dataset = load_recomp_dataset(args, split="train", context=args.use_context)
    # train_dataset = train_dataset[:10]
    # print(len(train_dataset))
    # print(train_dataset[:5])
    test_dataset = load_recomp_dataset(args, split="test", context=args.use_context)
    # test_dataset = test_dataset[:10]
    # print(len(test_dataset))
    # print(test_dataset[:5])
    # n_trains = int(len(train_dataset) * 0.8)
    # test_dataset = train_dataset[n_trains:]
    # train_dataset = train_dataset[:n_trains]
    train_dataset = RecompDataset(train_dataset, tokenizer, classification=False, causal=causal_model)
    test_dataset = RecompDataset(test_dataset, tokenizer, classification=False, causal=causal_model, test=True)
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
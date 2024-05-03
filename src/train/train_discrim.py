from utils.utils import get_model_tokenizer_cls, read_data
from utils.param import parse_args

from dataset.data import FluentDataset
from torch.utils.data import DataLoader
import torch

import torch.optim as optim
from transformers import default_data_collator, get_scheduler
from tqdm import tqdm

from sklearn.metrics import accuracy_score, mean_squared_error

import os

def process_labels(num_labels, items):
    output = []
    for item in items:
        item["score"] = min(item["score"], 4)
        item["score"] = max(item["score"], 1)
        if num_labels == 4:
            item["score"] = item["score"] - 1
        if num_labels == 2:
            item["score"] = int(item["score"] >= 3)
        output.append(item)
    return output

if __name__ == "__main__":
    args = parse_args()
    tokenizer, model = get_model_tokenizer_cls(args.base_model, args.num_labels)
    
    data_path = f"{args.root_path}/data/fluency/fluency_trainset.json"
    train_dataset = read_data(data_path)
    train_dataset = process_labels(args.num_labels, train_dataset)
    train_dataset = FluentDataset(train_dataset, tokenizer)
    
    data_path = f"{args.root_path}/data/fluency/fluency_valset.json"
    val_dataset = read_data(data_path)
    val_dataset = process_labels(args.num_labels, val_dataset)
    val_dataset = FluentDataset(val_dataset, tokenizer)

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
        with tqdm(total=len(val_loader)) as pbar:
            for i, batch in enumerate(val_loader):
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
                mse = mean_squared_error(labels, predictions)
                pbar.update(1)
                pbar.set_postfix(acc=acc, mse=mse)
        print(f"Accuracy for epoch {epoch}: {acc}")
        print(f"MSE for epoch {epoch}: {mse}")
    
    model_name = args.base_model.split("/")[-1]
    model_dir = f"{args.root_path}/save_models/discriminator/{model_name}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
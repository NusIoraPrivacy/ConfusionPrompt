from utils.param import parse_args
from utils.utils import get_model_tokenizer
from evaluation.gen_test_decomp import test_decomp

from dataset.data import DecompDataset
from dataset.get_data import *

from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import default_data_collator, get_scheduler

from tqdm import tqdm
import numpy as np
import os

if __name__ == '__main__':
    args = parse_args()
    tokenizer, model = get_model_tokenizer(args.base_model, args, device_map=args.device_map)
    # load train data
    decomp_inputs = load_decomp_dataset(args, split="train")
    # print(len(decomp_inputs))
    # decomp_inputs = decomp_inputs[:20]
    dataset = DecompDataset(decomp_inputs, tokenizer, args.token_len)
    dataloader = DataLoader(
            dataset, 
            batch_size=args.train_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=0.0,
                )
    num_training_steps = args.epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        )

    # train decompose model with supervised learning
    for epoch in range(args.epochs):
        model.train()
        loss_list = []

        with tqdm(
            total=int(len(dataset)/args.train_batch_size), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch'
        ) as pbar:

            for step, batch in enumerate(dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                output = model(**batch) 
                loss = output.loss
                loss_list.append(loss.item())
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            print(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')

        if (epoch+1) % args.save_epo == 0 or epoch+1 == args.epochs:
            model_name = args.base_model.split("/")[-1]
            model_dir = f"{args.root_path}/save_models/decomp/{args.test_mode}/{args.decomp_data}/{model_name}/epoch_{epoch+1}"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
            if args.use_peft:
                model = model.merge_and_unload()
            model.save_pretrained(model_dir)
        
        if args.val:
            model.eval()
            print(f"Test for epoch {epoch+1}")
            test_decomp(args, model, tokenizer, epoch+1)
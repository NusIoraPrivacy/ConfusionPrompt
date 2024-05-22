from utils.param import parse_args
from utils.utils import get_model_tokenizer
from dataset.data import ReplaceDataset
from dataset.get_data import get_p2f
import json
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
import torch.optim as optim
import numpy as np
from evaluation.gen_test_replace import test_replace, test_replace_p2f

if __name__ == '__main__':
    args = parse_args()
    
    tokenizer, model = get_model_tokenizer(args.base_model, args)
    # load train data
    if args.decomp_data == "p2f":
        replace_inputs = get_p2f(args, split="train")
        # replace_inputs = replace_inputs[:10]
        
    else:
        data_path = f"{args.root_path}/data/{args.decomp_data}/replace_train_samples.json"
        replace_inputs = []
        with open(data_path) as dataset_file:
            for line in dataset_file:
                this_item = json.loads(line)
                replace_inputs.append(this_item) 
        data_path = f"{args.root_path}/results/{args.decomp_data}/replace/replace_candidates.json"
        with open(data_path) as f:
            this_inputs = json.load(f)
        for sample in this_inputs:
            question, attributes, replacements = sample["question"], sample["attributes"], sample["replace sentences"]
            try:
                replacements = eval(replacements)
            except Exception as e:
                continue
            for replacement in replacements:
                this_item = {"raw query": question, "attributes": attributes, "replaced query": replacement}
                replace_inputs.append(this_item)
    # replace_inputs = replace_inputs[:10]
    rpl_dataset = ReplaceDataset(replace_inputs, tokenizer)

    dataloader = DataLoader(
            rpl_dataset, 
            batch_size=args.train_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    # weight_decay=0.0,
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
            total=int(len(rpl_dataset)/args.train_batch_size), desc=f'Epoch {epoch + 1}/{args.epochs}', unit='batch'
        ) as pbar:

            for step, batch in enumerate(dataloader):
                # just query gpt when epoch == 0
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

        # if (epoch+1) % args.save_epo == 0 or epoch+1 == args.epochs:
        #     model_name = args.base_model.split("/")[-1]
        #     model_dir = f"{args.root_path}/save_models/replace/{args.decomp_data}/{model_name}/epoch_{epoch+1}"
        #     if not os.path.exists(model_dir):
        #         os.makedirs(model_dir, exist_ok=True)
        #     model.save_pretrained(model_dir)
        
        if args.val:
            model.eval()
            print(f"Test for epoch {epoch}")
            if args.decomp_data == "p2f":
                test_replace_p2f(args, model, tokenizer, epoch+1)
            else:
                test_replace(args, model, tokenizer, epoch+1)

    model_name = args.base_model.split("/")[-1]
    model_dir = f"{args.root_path}/save_models/replace/{args.decomp_data}/sp/{model_name}/final"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
from utils.param import parse_args
from utils.utils import load_bsldataset, write_list, read_data, get_model_tokenizer, get_eval_model
from utils.globals import *
from baselines.bsl_utils import *

from models.gpt import *

import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, pipeline

from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import accuracy_score, roc_auc_score
import os

def sample_noise_Gaussian(d_shape, sensitivity, args):
    sigma = np.sqrt(2* np.log(1.25/args.delta)) / args.epsilon
    noise = torch.normal(mean=0., std=sigma*sensitivity, size=d_shape)
    return noise

def get_token_embedding(token_id, model, model_name):
    """get the token embedding given the input ids"""
    with torch.no_grad():
        if model_name =="stevhliu/my_awesome_model":
            embeddings = model.distilbert.embeddings.word_embeddings(token_id)
            # embeddings = model.distilbert.embeddings(token_id)
        elif model_name in ("bert-base-uncased", "bert-large-uncased"):
            embeddings = model.bert.embeddings.word_embeddings(token_id)
        elif 'gpt2' in model_name:
            embeddings = model.wte(token_id)
        elif 'opt' in model_name:
            try:
                embeddings = model.model.decoder.embed_tokens(token_id)
            except:
                embeddings = model.decoder.embed_tokens(token_id)
        elif 'llama' in model_name:
            try:
                embeddings = model.model.embed_tokens(token_id)
            except:
                embeddings = model.embed_tokens(token_id)
        elif 't5' in model_name:
            embeddings = model.encoder.embed_tokens(token_id)
        elif "bart" in model_name:
            embeddings = model.model.shared.weight[token_id]
        embeddings = embeddings.squeeze()
    return embeddings

def get_closest_token(embedding, tokenizer, model, model_name):
    """Find the word with the closest embedding."""
    closest_token = None
    if 'gpt2' in model_name:
        vocabulary = tokenizer.get_vocab()
    else:
        vocabulary = tokenizer.vocab
    token_ids = [token_id for _, token_id in vocabulary.items()]
    token_ids = torch.tensor(token_ids).to(embedding.device)
    word_embeddings = get_token_embedding(token_ids, model, model_name)
    embedding = embedding.unsqueeze(dim=0)
    embedding = embedding.expand(word_embeddings.size())
    distance = torch.norm(embedding - word_embeddings, 2, dim=1)
    closest_idx = distance.argmin()
    closest_token = token_ids[closest_idx]
    return closest_token.item()

def text2text_priv(input_id, tokenizer, model, args):
    init_embeddings = get_token_embedding(input_id, model, args.t2t_model)
    # clip embedding
    max_norm = emb_norm_dict[args.t2t_model]
    all_norms = torch.norm(init_embeddings, p=2, dim=-1)
    init_embeddings = init_embeddings * torch.clamp(max_norm / all_norms, max=1).unsqueeze(-1)
    # sample noise
    noises = sample_noise_Gaussian(init_embeddings.shape, max_norm, args)
    noises = noises.to(init_embeddings.device)
    init_embeddings = init_embeddings + noises
    # acc = 0
    for i in range(1, len(init_embeddings)-1):
        embed = init_embeddings[i]
        closest_token = get_closest_token(embed, tokenizer, model, args.t2t_model)
        # if closest_token == input_id[i]:
        #     acc += 1
        input_id[i] = closest_token
    # acc /= (len(init_embeddings) - 2)
    return input_id

if __name__ == "__main__":
    args = parse_args()
    val_dataset = load_bsldataset(args)
    t2t_tokenizer, t2t_model = get_model_tokenizer(args.t2t_model, args)
    # apply text2text privatization
    privatized_dataset = []
    for sample in tqdm(val_dataset):
        question = sample["question"]
        # print(f"Original question: {question}")
        input_ids = t2t_tokenizer.encode(question, return_tensors='pt').squeeze().to(t2t_model.device)
        input_ids = text2text_priv(input_ids, t2t_tokenizer, t2t_model, args)
        sent = t2t_tokenizer.decode(token_ids = input_ids, skip_special_tokens=True)
        # print(f"Privatized question: {sent}")
        sample["question"] = sent
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
    output_path = f"{output_dir}/predictions_{args.eval_model}_t2t.json"
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
    
    data_path = f"{args.root_path}/results/{args.eval_data}/baseline/predictions_{args.eval_model}_t2t.json"
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
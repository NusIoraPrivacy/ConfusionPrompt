from utils.globals import *
from utils.param import str2bool
from utils.score_utils import *
from utils.utils import get_model_tokenizer_qa, get_model_tokenizer_cls

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
from sentence_transformers import SentenceTransformer, util

from dataset.data import ResctructCPDataset, AttrCPDataset
from sklearn import metrics

from tqdm import tqdm
import random
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
    parser.add_argument("--mu", type=float, default=0.1, help="privacy budget")
    parser.add_argument("--sim_thd", type=float, default=0.7, help = "threshold for similarity")
    parser.add_argument("--flu_thd", type=int, default=3, help = "threshold for fluency")
    parser.add_argument("--attack_type", type=str, default="reconstruct", choices=["reconstruct", "attribute"])
    parser.add_argument("--num_labels", type=int, default=4, choices=[2, 4])
    parser.add_argument("--disriminator", type=str, default="bert-base-uncased")
    parser.add_argument("--attr_extractor", type=str, default="facebook/bart-large")
    parser.add_argument("--reconstruct_mode", type=str, default="generate", choices=["select", "generate"])
    parser.add_argument("--attr_ratio", type=int, default=1, help = "the ratio between false and true attributes")
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    return args

def get_qualify_inputs(inputs, flu_tokenizer, flu_model, attr_tokenizer, attr_model, sim_model, args):
    out_data = []

    for item in tqdm(inputs):
        question, attributes, rpl_sents = item["question"], item["attributes"], item["replace sentences"]
        try:
            rpl_sents = eval(rpl_sents)
        except Exception as e:
            continue

        qualified_rpt_sents = []
        qualified_rpl_phrases = []
        for rpl_sent in rpl_sents:
            # evaluate sentence fluency
            rpl_inputs = flu_tokenizer(rpl_sent, return_tensors="pt")
            for key in rpl_inputs :
                rpl_inputs[key] = rpl_inputs[key].to(flu_model.device)
            with torch.no_grad():
                outputs = flu_model(**rpl_inputs)
            logits = outputs.logits
            y_pred = torch.argmax(logits, -1)
            flu_score = y_pred[0].item()
            if flu_score < args.flu_thd:
                continue
            
            # evaluate attribute similarity
            phrases = []
            for attr in attributes:
                sim_prompt = f"{question} {attr_tokenizer.bos_token} {rpl_sent} {attr_tokenizer.bos_token} {attr}"
                input_id = torch.tensor(
                    attr_tokenizer.encode(sim_prompt), dtype=torch.int64
                )
                input_id = input_id.unsqueeze(0)
                input_id = input_id.to(attr_model.device)
                phrase_ids = attr_model.generate(input_id, max_new_tokens=50)
                this_phrase = attr_tokenizer.batch_decode(phrase_ids, skip_special_tokens=True)[0]
                this_phrase = this_phrase.replace(sim_prompt, "")
                phrases.append(this_phrase)
            # print(question)
            # print(attributes)
            # print(rpl_sent)
            # print(phrases)
            max_sim = 0
            for attr, rpl_attr in zip(attributes, phrases):
                emb1 = sim_model.encode(attr)
                emb2 = sim_model.encode(rpl_attr)
                cos_sim = util.cos_sim(emb1, emb2)
                cos_sim = cos_sim[0][0].item()
                max_sim = max(max_sim, cos_sim)
            if max_sim > args.sim_thd:
                continue
            qualified_rpt_sents.append(rpl_sent)
            qualified_rpl_phrases.append(phrases)
            if len(qualified_rpt_sents) >= int(1/args.mu) * len(attributes):
                break

        if len(qualified_rpt_sents) >= int(1/args.mu):
            start = 0
            interval = int(len(qualified_rpt_sents) / len(attributes))
            for i, attr in enumerate(attributes):
                this_rpt_sents = qualified_rpt_sents[start:start+int(1/args.mu)]
                if args.attack_type == "reconstruct":
                    out_data.append({"question": question, "attribute": attr, "replace sentences": this_rpt_sents})
                else:
                    all_rpl_attrs = [phrase[i] for phrase in qualified_rpl_phrases]
                    random.shuffle(all_rpl_attrs)
                    selected_rpl_attrs = all_rpl_attrs[:args.attr_ratio]
                    out_data.append({"question": question, "attribute": attr, "replace sentences": this_rpt_sents, 
                    "replace attributes": selected_rpl_attrs})
                start += interval
    return out_data

def init_optim(model, train_loader, args):
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
    return optimizer, lr_scheduler

if __name__ == "__main__":
    args = parse_args()

    data_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_candidates.json"
    with open(data_path) as f:
        inputs = json.load(f)
    if args.debug:
        inputs = inputs[:20]
        args.epochs = 3
    model_name = args.disriminator.split("/")[-1]
    model_dir =  f"{_ROOT_PATH}/save_models/discriminator/{model_name}"
    flu_tokenizer, flu_model = get_model_tokenizer_cls(model_dir, args.num_labels)
    model_name = args.attr_extractor.split("/")[-1]
    model_dir = f"{_ROOT_PATH}/save_models/attr_extractor/{model_name}"
    attr_tokenizer, attr_model = get_model_tokenizer_qa(model_dir)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    n_trains = int(len(inputs)*0.8)
    train_inputs = inputs[:n_trains]
    test_inputs = inputs[n_trains:]
    # for each item, filter the replacement satisfying the similairy and fluency criteria   
    train_data = get_qualify_inputs(train_inputs, flu_tokenizer, flu_model, attr_tokenizer, attr_model, sim_model, args)
    test_data = get_qualify_inputs(test_inputs, flu_tokenizer, flu_model, attr_tokenizer, attr_model, sim_model, args)

    # train a model to conduct the attack
    if args.attack_type == "reconstruct":
        if args.reconstruct_mode == "select":
            num_labels = int(1/args.mu) + 1
            tokenizer, model = get_model_tokenizer_cls(args.model, num_labels)
        
        else:
            tokenizer, model = get_model_tokenizer_qa(args.model)
        
        train_dataset = ResctructCPDataset(train_data, tokenizer, mode=args.reconstruct_mode)
        test_dataset = ResctructCPDataset(test_data, tokenizer, mode=args.reconstruct_mode)
        train_loader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=args.train_batch_size)
        test_loader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.test_batch_size)
        optimizer, lr_scheduler = init_optim(model, train_loader, args)

        best_scores = {"roughL": 0, "blue": 0, "f1": 0, "exact match": 0}
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
                    if args.reconstruct_mode == "select":
                        with torch.no_grad():
                            outputs = model(
                                    input_ids = batch["input_ids"], 
                                    attention_mask = batch["attention_mask"])
                            logits = outputs.logits
                            y_preds = torch.argmax(logits, -1)
                        this_labels = []
                        this_predictions = []
                        for input_id, true_idx, pred_idx in zip(batch["input_ids"], batch["labels"], y_preds):
                            prompt = tokenizer.decode(input_id)
                            prompt = prompt.strip(tokenizer.pad_token)
                            all_queries = prompt.split(tokenizer.bos_token)
                            true_query = all_queries[true_idx]
                            pred_query = all_queries[pred_idx]
                            this_labels.append(true_query)
                            this_predictions.append(pred_query)
                        labels += this_labels
                        predictions += this_predictions
                    else:
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
                        # print(y_preds)

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
                    pbar.set_postfix(rougnL=rougnL, blue=_blue, f1=f1, exact_match=exact_match)
                    # for prompt, y_pred, label in zip(prompts, y_preds, this_labels):
                    #     outputs.append({"prompt": prompt, "prediction": y_pred, "label": label})
            print(f"RougnL for epoch {epoch}: {rougnL}")
            print(f"Blue for epoch {epoch}: {_blue}")
            print(f"F1 for epoch {epoch}: {f1}")
            print(f"Exact match for epoch {epoch}: {exact_match}")
            if best_scores["blue"] < _blue:
                best_scores["roughL"] = rougnL
                best_scores["blue"] = _blue
                best_scores["f1"] = f1
                best_scores["exact match"] = exact_match
            
        print("best results: ")
        print(best_scores)
    
    if args.attack_type == "attribute":
        num_labels = 2
        tokenizer, model = get_model_tokenizer_cls(args.model, num_labels)

        train_dataset = AttrCPDataset(train_data, tokenizer)
        test_dataset = AttrCPDataset(test_data, tokenizer)
        train_loader = DataLoader(train_dataset, collate_fn=default_data_collator, batch_size=args.train_batch_size)
        test_loader = DataLoader(test_dataset, collate_fn=default_data_collator, batch_size=args.train_batch_size)
        optimizer, lr_scheduler = init_optim(model, train_loader, args)

        best_scores = {"f1": 0, "precision": 0, "recall": 0, "accuracy": 0}
        best_f1 = 0
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
            with tqdm(total=len(test_loader)) as pbar:
                for i, batch in enumerate(test_loader):
                    for key in batch:
                        batch[key] = batch[key].to(model.device)
                        with torch.no_grad():
                            outputs = model(
                                    input_ids = batch["input_ids"], 
                                    attention_mask = batch["attention_mask"])
                            logits = outputs.logits
                            y_preds = torch.argmax(logits, -1)
                        predictions += y_preds.tolist()
                        labels += batch["labels"].tolist()
                        precision = metrics.precision_score(labels, predictions)
                        recall = metrics.recall_score(labels, predictions)
                        f1 = metrics.f1_score(labels, predictions)
                        acc = metrics.accuracy_score(labels, predictions)

                        pbar.update(1)
                        pbar.set_postfix(precision=precision, recall=recall, f1=f1, acc=acc)
            # print(labels)
            # print(predictions)
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
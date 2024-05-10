from utils.globals import *
from utils.param import str2bool
from utils.score_utils import *
from utils.utils import get_model_tokenizer_qa, get_model_tokenizer_cls, get_model_tokenizer, get_eval_model
from baselines.text2text import text2text_priv

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
from sentence_transformers import SentenceTransformer, util

from dataset.data import ResctructDataset, AttrDataset
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
    parser.add_argument("--para_model", type=str, default="eugenesiow/bart-paraphrase",
                        choices=["google/flan-t5-xl", "eugenesiow/bart-paraphrase"])
    parser.add_argument("--t2t_model", type=str, default="facebook/bart-large")
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
    parser.add_argument("--delta", type=float, default=1e-4, help="privacy budget")
    parser.add_argument("--epsilon", type=float, default=10, help="privacy budget")
    parser.add_argument("--dp_type", type=str, default="text2text", choices=["text2text", "paraphrase"])
    parser.add_argument("--attack_type", type=str, default="attribute", choices=["reconstruct", "attribute"])
    parser.add_argument("--gen_sample", type=str2bool, default=True)
    parser.add_argument("--gen_replacement", type=str2bool, default=False)
    parser.add_argument("--attack_flag", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.gen_replacement:
        gpt_model = get_eval_model(args, "gpt-4-turbo")
        data_path = f"{_ROOT_PATH}/results/{args.data_name}/replace/question_attrs.json"
        output_path = f"{_ROOT_PATH}/results/{args.data_name}/replace/question_attrs_replace.json"
        with open(data_path) as f:
            dataset = json.load(f)
        output = {}
        for question, attributes in tqdm(dataset.items()):
            prompt = replace_template_phrase.format(sentence=question, attributes=attributes)
            prompts = [prompt]
            # print(prompts)
            success = False
            n_tries = 0
            while not success:
                rpl_phrases = gpt_model.generate(prompts)[0]
                try:
                    rpl_phrases = eval(rpl_phrases)
                    if len(rpl_phrases) == len(attributes):
                        success = True
                    else:
                        n_tries += 1
                except Exception as e:
                    n_tries += 1
                if n_tries >= 5:
                    break
            if success:
                output[question] = {"attributes": attributes, "replace attributes": rpl_phrases}
                with open(output_path, "w") as f:
                    json.dump(output, f, indent=4)

    if args.gen_sample:
        data_path = f"{_ROOT_PATH}/results/{args.data_name}/replace/question_attrs_replace.json"
        with open(data_path) as f:
            dataset = json.load(f)
        if args.debug:
            temp = {}
            for i, key in enumerate(dataset):
                if i >= 20:
                    break
                temp[key] = dataset[key]
                
            dataset = temp
        if args.dp_type == "paraphrase":
            para_tokenizer, para_model = get_model_tokenizer_qa(args.para_model)
            privatized_dataset = []
            for question, attr_dict in tqdm(dataset.items()):
                if "paraphrase" not in args.para_model:
                    prompt = f"Please paraphrase the following question: {question}"
                else:
                    prompt = question
                input_ids = para_tokenizer.encode(prompt, return_tensors='pt').to(para_model.device)
                n = int(1.2 * len(input_ids[0]))
                cnt = 0
                lower_bound = logit_range_dict[args.para_model][0]
                upper_bound = logit_range_dict[args.para_model][1]
                outputs = para_model.generate(input_ids, do_sample=True, temperature = (2*(upper_bound-lower_bound)*n/args.epsilon), max_new_tokens=n)
                # outputs = para_model.generate(input_ids, do_sample=True, temperature = 1, max_new_tokens=n)
                para_question = para_tokenizer.decode(token_ids = outputs[0], skip_special_tokens=True)
                para_question = para_question.replace(prompt, "")
                # print("original quesiont:", question)
                # print("paraphrase question:", para_question)
                this_item = {"question": question, "attributes": attr_dict["attributes"], 
                            "privatized question": para_question, "replace attributes": attr_dict["replace attributes"]}
                privatized_dataset.append(this_item)
            model_name = args.para_model.split("/")[-1]
            output_path = f"{_ROOT_PATH}/results/{args.data_name}/attack/dp/para_{model_name}_questions_eps_{args.epsilon}.json"
            with open(output_path, "w") as f:
                json.dump(privatized_dataset, f, indent=4)
        
        else:
            t2t_tokenizer, t2t_model = get_model_tokenizer(args.t2t_model)
            privatized_dataset = []
            for question, attr_dict in tqdm(dataset.items()):
                # print(f"Original question: {question}")
                input_ids = t2t_tokenizer.encode(question, return_tensors='pt').squeeze().to(t2t_model.device)
                input_ids = text2text_priv(input_ids, t2t_tokenizer, t2t_model, args)
                t2t_question = t2t_tokenizer.decode(token_ids = input_ids, skip_special_tokens=True)
                # print(f"Privatized question: {sent}")
                this_item = {"question": question, "attributes": attr_dict["attributes"],  
                "privatized question": t2t_question, "replace attributes": attr_dict["replace attributes"]}
                privatized_dataset.append(this_item)
            model_name = args.t2t_model.split("/")[-1]
            output_path = f"{_ROOT_PATH}/results/{args.data_name}/attack/dp/t2t_{model_name}_questions_eps_{args.epsilon}.json"
            with open(output_path, "w") as f:
                json.dump(privatized_dataset, f, indent=4)
    
    if args.attack_flag:
        prefix = "para" if args.dp_type == "paraphrase" else "t2t"
        if args.dp_type == "paraphrase":
            model_name = args.para_model.split("/")[-1]
        else:
            model_name = args.t2t_model.split("/")[-1]
        data_path = f"{_ROOT_PATH}/results/{args.data_name}/attack/dp/{prefix}_{model_name}_questions_eps_{args.epsilon}.json"
        with open(data_path) as f:
            dataset = json.load(f)
        if args.debug:
            dataset = dataset[:20]
            args.epochs = 5
        if args.attack_type == "reconstruct":
            tokenizer, model = get_model_tokenizer_qa(args.model)
        else:
            tokenizer, model = get_model_tokenizer_cls(args.model, 2)
        n_trains = int(len(dataset) * 0.8)
        train_data = dataset[:n_trains]
        test_data = dataset[n_trains:]
        if args.attack_type == "reconstruct":
            train_dataset = ResctructDataset(train_data, tokenizer)
            test_dataset = ResctructDataset(test_data, tokenizer)
        else:
            train_dataset = AttrDataset(train_data, tokenizer)
            test_dataset = AttrDataset(test_data, tokenizer)

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
        if args.attack_type == "reconstruct":
            best_scores = {"roughL": 0, "blue": 0, "f1": 0, "exact match": 0}
        else:
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
                    if args.attack_type == "reconstruct":
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
                        pbar.set_postfix(rougnL=rougnL, blue=_blue, f1=f1, exact_match=exact_match)

                    else:
                        with torch.no_grad():
                            outputs = model(
                                    input_ids = batch["input_ids"], 
                                    attention_mask = batch["attention_mask"])
                        logits = outputs.logits
                        y_preds = torch.argmax(logits, -1).tolist()
                        predictions += y_preds
                        labels += batch["labels"].tolist()
                        
                        precision = metrics.precision_score(labels, predictions)
                        recall = metrics.recall_score(labels, predictions)
                        f1 = metrics.f1_score(labels, predictions)
                        acc = metrics.accuracy_score(labels, predictions)

                        pbar.update(1)
                        pbar.set_postfix(precision=precision, recall=recall, f1=f1, acc=acc)


                    
                    # for prompt, y_pred, label in zip(prompts, y_preds, this_labels):
                    #     outputs.append({"prompt": prompt, "prediction": y_pred, "label": label})
            if args.attack_type == "reconstruct":
                print(f"RougnL for epoch {epoch}: {rougnL}")
                print(f"Blue for epoch {epoch}: {_blue}")
                print(f"F1 for epoch {epoch}: {f1}")
                print(f"Exact match for epoch {epoch}: {exact_match}")
                if best_scores["blue"] < _blue:
                    best_scores["roughL"] = rougnL
                    best_scores["blue"] = _blue
                    best_scores["f1"] = f1
                    best_scores["exact match"] = exact_match
            else:
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
from utils.param import parse_args
from utils.utils import get_eval_model, write_list, read_data
from utils.globals import *
from utils.score_utils import *
from baselines.bsl_utils import *

from dataset.data import BslData
from dataset.get_data import load_bsldataset
from models.gpt import *

from torch.utils.data import DataLoader
from transformers import default_data_collator, pipeline

from tqdm import tqdm
import numpy as np
import json
from sklearn.metrics import accuracy_score, roc_auc_score
import os
import re

def extract_prediction(pred):
    start_idxs = [i.start(0) for i in re.finditer("(C|c)onclusion", pred)]
    if len(start_idxs) > 0:
        start_idx = start_idxs[0]
        pred = pred[start_idx:]
        pred = re.sub('(C|c)onclusion(:?)', '', pred)
    return pred

if __name__ == "__main__":
    args = parse_args()
    val_dataset = load_bsldataset(args)
    val_dataset = BslData(val_dataset, args)
    val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.test_batch_size, 
            collate_fn=seq_collate_fn, 
            pin_memory=True,
            shuffle=True
        )
    
    lbl_dict = {
        "strategyQA": ["No", "Yes"],
        "hotpotqa-yn": ["No", "Yes"],
        }

    suffix = "cot" if args.step_by_step else "dq"
    output_dir = f"{args.root_path}/results/{args.eval_data}/baseline"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    model_name = args.eval_model.split("/")[-1]
    output_path = f"{output_dir}/predictions_{model_name}_{suffix}.json"

    data_type = dataset_type[args.eval_data]
    if data_type == "cls":
        candidate_labels = lbl_dict[args.eval_data]
    if "gpt" not in args.eval_model:
        if data_type == "cls":
            oracle = pipeline(task="zero-shot-classification", 
                    model=args.eval_model, device_map="auto")
        elif data_type == "qa":
            oracle = pipeline(task="text-generation", 
                model=args.eval_model, device_map="auto")

        labels = []
        predictions = []
        output = []
        max_step = 500
        with tqdm(total=min(len(val_dataloader), max_step), unit='batch') as pbar:
            for step, batch in enumerate(val_dataloader):
                # if data_type == "cls":
                #     results = oracle(
                #         sequences = batch["sequences"],
                #         candidate_labels = candidate_labels,
                #     )
                # elif data_type == "qa":
                #     results = oracle(
                #         text_inputs = batch["sequences"]
                #     )
                results = oracle(
                        text_inputs = batch["sequences"], max_new_tokens=args.max_new_tokens
                    )
                this_labels = []
                this_preds = []
                for i, rslt in enumerate(results):
                    # if data_type == "cls":
                    #     label, score = rslt['labels'], rslt['scores']
                    #     max_idx = np.argmax(score)
                    #     max_label = label[max_idx]
                    #     pred = candidate_labels.index(max_label)
                    # elif data_type == "qa":
                    #     question = batch["sequences"][i]
                    #     # print(question)
                    #     pred = rslt[0]['generated_text']
                    #     pred = pred.replace(question, "")
                    #     # print(pred)
                    #     if args.step_by_step:
                    #         pred = extract_prediction(pred)
                    question = batch["sequences"][i]
                    # print(question)
                    pred = rslt[0]['generated_text']
                    pred = pred.replace(question, "")
                    # print(pred)
                    if args.step_by_step:
                        pred = extract_prediction(pred)
                    if data_type == "cls":
                        pred = standard_ans(pred)
                    predictions.append(pred)
                    this_preds.append(pred)

                    for label in batch['labels']:
                        if data_type == "cls":
                            label = standard_ans(label)
                        labels.append(label)
                        this_labels.append(label)
                    # print(pred)
                    # print(label)
                if data_type == "cls":
                    acc = accuracy_score(labels, predictions)
                    auc = roc_auc_score(labels, predictions)
                    pbar.set_postfix(acc=acc, auc=auc)
                elif data_type == "qa":
                    rougnL = rouge(predictions, labels)
                    f1 = f1_score(predictions, labels)
                    exact_match = exact_match_score(predictions, labels)
                    pbar.set_postfix(rougnL=rougnL, f1=f1, exact_match=exact_match)
                questions = batch["questions"]
                for question, label, pred in zip(questions, this_labels, this_preds):
                    item = {"question": question,
                            "label": label,
                            "prediction": pred}
                    output.append(item)
                write_list(output_path, output)
                pbar.update(1)
                if step >= max_step:
                    break
        
        if data_type == "cls":
            print(f"Accuracy: {acc}")
            print(f"AUC: {auc}")
        elif data_type == "qa":
            print(f"RougnL: {rougnL}")
            print(f"F1: {f1}")
            print(f"Exact match: {exact_match}")
    
    else:
        model = get_eval_model(args)
        output = []
        max_step = 500
        with tqdm(total=min(len(val_dataloader), max_step), unit='batch') as pbar:
            for step, batch in enumerate(val_dataloader):
                prompts = batch["sequences"]
                labels = batch["labels"]
                questions = batch["questions"]
                preds = model.generate(prompts)
                pbar.update(1)
                for question, label, raw_pred in zip(questions, labels, preds):
                    if data_type == "cls":
                        label = candidate_labels[label]
                    if args.step_by_step:
                        pred = extract_prediction(raw_pred)
                    else:
                        pred = raw_pred
                    item = {"question": question,
                            "label": label,
                            "raw_answer": raw_pred,
                            "prediction": pred}
                    output.append(item)
                write_list(output_path, output)
                if step >= max_step:
                    break
        
        data_path = f"{args.root_path}/results/{args.eval_data}/baseline/predictions_{args.eval_model}_{suffix}.json"
        results = read_data(data_path)
        labels = []
        predictions = []
        for item in results:
            label = item["label"].lower()
            pred = item["prediction"]
            if data_type == "cls":
                label = standard_ans(label)
                pred = standard_ans(pred)
            labels.append(label)
            predictions.append(pred)
        if data_type == "cls":
            acc = accuracy_score(labels, predictions)
            auc = roc_auc_score(labels, predictions)
            print(f"Accuracy: {acc}")
            print(f"AUC: {auc}")
        elif data_type == "qa":
            rougnL = rouge(predictions, labels)
            f1 = f1_score(predictions, labels)
            exact_match = exact_match_score(predictions, labels)
            print(f"RougnL: {rougnL}")
            print(f"F1: {f1}")
            print(f"Exact match: {exact_match}")
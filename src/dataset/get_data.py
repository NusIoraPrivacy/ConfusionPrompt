from utils.globals import *

import json

from datasets import load_dataset

def load_decomp_dataset(args, split="train"):
    data_file = decomp_dict[args.decomp_data][split]
    data_path = f"{args.root_path}/data/{args.decomp_data}/{data_file}"
    if args.decomp_data in ("strategyQA", "hotpotqa-yn") or (args.decomp_data == "hotpotqa" and split=="test"):
        with open(data_path) as dataset_file:
            dataset = json.load(dataset_file)
    elif args.decomp_data=="musique" or (args.decomp_data == "hotpotqa" and split=="train"):
        with open(data_path) as dataset_file:
            dataset = []
            for line in dataset_file:
                this_item = json.loads(line)
                if args.decomp_data == "hotpotqa" and this_item["f1"] <= 0.6:
                    continue
                dataset.append(this_item)
    
    decomp_items = []
    for item in dataset:
        question = item[decomp_dict[args.decomp_data]["question"]]
        if args.decomp_data == "musique":
            temp = []
            for decomp in decomposition:
                temp.append(decomp["question"])
            decomposition = temp
        else:
            try:
                decomposition = item[decomp_dict[args.decomp_data]["decomp"]]
            except Exception as e:
                decomposition = ""
        this_item = {
            "question": question,
            "decomposition": decomposition
            }
        decomp_items.append(this_item)
        
    return decomp_items

def load_recomp_dataset(args, split="train"):
    data_file = decomp_dict[args.recomp_data][split]
    data_path = f"{args.root_path}/data/{args.recomp_data}/{data_file}"
    if args.recomp_data in ("strategyQA", "hotpotqa-yn")  or (args.recomp_data == "hotpotqa" and split=="test"):
        with open(data_path) as dataset_file:
            dataset = json.load(dataset_file)
    elif args.recomp_data=="musique" or (args.recomp_data == "hotpotqa" and split=="train"):
        with open(data_path) as dataset_file:
            dataset = []
            for line in dataset_file:
                this_item = json.loads(line)
                if args.recomp_data == "hotpotqa" and this_item["f1"] <= 0.6:
                    continue
                dataset.append(this_item)
    
    recomp_items = []
    # levels = []
    for item in dataset:
        # obtain question
        question = item[decomp_dict[args.decomp_data]["question"]]
        # obtain decomposition
        if args.recomp_data == "musique":
            temp = []
            for decomp in decomposition:
                temp.append(decomp["question"])
            decomposition = temp
        else:
            try:
                decomposition = item[decomp_dict[args.decomp_data]["decomp"]]
            except Exception as e:
                decomposition = ""
        # obtain sub-answer and facts
        if args.recomp_data == "strategyQA":
            sub_answer = item["facts"]
            label = 1 if item["answer"] == True else 0
        elif args.recomp_data == "hotpotqa-yn":
            try:
                sub_answer = item["subanswers"]
            except Exception as e:
                sub_answer = ""
            label = 1 if item["answer"].lower() == "yes" else 0
        elif args.recomp_data == "hotpotqa":
            try:
                sub_answer = item["subanswers"]
            except Exception as e:
                sub_answer = ""
            try:
                label = item["answer"]
            except Exception as e:
                label = item["label"]
        this_item = {
            "question": question,
            "decomposition": decomposition,
            "sub-answers": sub_answer,
            "label": label
            }
        recomp_items.append(this_item)
        # levels.append(item["level"])
        
    return recomp_items

def load_bsldataset(args):
    data_file = decomp_dict[args.eval_data]["test"]
    data_path = f"{args.root_path}/data/{args.eval_data}/{data_file}"
    with open(data_path) as f:
        val_dataset = json.load(f)
    output = []
    for sample in val_dataset:
        if args.eval_data == "strategyQA":
            item = {
                "question": sample["question"],
                "label": 1 if sample["answer"] == True else 0
            }
        elif args.eval_data == "hotpotqa-yn":
            item = {
                "question": sample["question"],
                "label": 1 if sample["answer"].lower() == "yes" else 0
            }
        elif args.eval_data == "hotpotqa":
            item = {
                "question": sample["question"],
                "label": sample["answer"]
            }
        output.append(item)
    return output

def load_twentyquery(args, split="train", binary=True):
    mode = "train" if split == "train" else "dev"
    data_path = f"{args.root_path}/data/twentyquestions/twentyquestions-{mode}.jsonl"
    dataset = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            subject = item["subject"]
            question = item["question"]
            original_question = question
            question = question.replace(" it ", " " + subject + " ")
            question = question.replace(" it?", " " + subject)
            if question.endswith(" it"):
                question = question[: -len("it")] + subject
            if question.startswith("it "):
                question = subject + question[len("it")]
            if original_question != question:
                question = question.replace("?", "")
                question = question.replace("\t", "")
            question = question if question.endswith("?") else question + "?"
            if binary:
                answer = int(item["majority"])
            else:
                answer = "yes" if item["majority"] else "no"
            dataset.append({"question": question, "context": "", "label": answer})
    return dataset

def load_boolq(args, split="train", binary=True):
    mode = "train" if split == "train" else "dev"
    data_path = f"{args.root_path}/data/boolq/{mode}.jsonl"
    dataset = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            question = item["question"]
            question = question if question.endswith("?") else question + "?"
            context = item["passage"]
            if binary:
                answer = int(item["answer"])
            else:
                answer = "yes" if item["majority"] else "no"
            dataset.append({"question": question, "context": context, "label": answer})
    return dataset

def get_drop(args, split="train"):
    split = "train" if split == "train" else "validation"
    raw_dataset = load_dataset("ucinlp/drop", split=split)
    questions = raw_dataset["question"]
    contexts = raw_dataset["passage"]
    answers = [item["spans"][0] for item in raw_dataset["answers_spans"]]
    output = []
    for question, context, answer in zip(questions, contexts, answers):
        output.append({"question": question, "context": context, "label": answer})
    return output

def get_squad(args, split="train"):
    split = "train" if split == "train" else "validation"
    raw_dataset = load_dataset("rajpurkar/squad", split=split)
    questions = raw_dataset["question"]
    contexts = raw_dataset["context"]
    answers = [item["text"][0] for item in raw_dataset["answers"]]
    output = []
    for question, context, answer in zip(questions, contexts, answers):
        output.append({"question": question, "context": context, "label": answer})
    return output
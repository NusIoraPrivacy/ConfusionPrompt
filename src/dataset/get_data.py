from utils.globals import *
from utils.utils import format_question

import json

from datasets import load_dataset
from tqdm import tqdm

def load_decomp_dataset(args, split="train"):
    data_file = decomp_dict[args.decomp_data][split]
    data_path = f"{args.root_path}/data/{args.decomp_data}/{data_file}"
    if args.decomp_data=="musique" or (args.decomp_data == "hotpotqa-gpt" and split=="train"):
        with open(data_path) as dataset_file:
            dataset = []
            for line in dataset_file:
                this_item = json.loads(line)
                if args.decomp_data == "hotpotqa" and this_item["f1"] <= 0.6:
                    continue
                dataset.append(this_item)
    else:
        with open(data_path) as dataset_file:
            dataset = json.load(dataset_file)
    
    decomp_items = []
    for item in dataset:
        question = item[decomp_dict[args.decomp_data]["question"]]
        if args.decomp_data == "musique":
            temp = []
            decomposition = item[decomp_dict[args.decomp_data]["decomp"]]
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

def load_recomp_dataset(args, split="train", context=False):
    data_file = decomp_dict[args.recomp_data][split]
    data_path = f"{args.root_path}/data/{args.recomp_data}/{data_file}"
    if args.recomp_data=="musique" or (args.recomp_data == "hotpotqa-gpt" and split=="train"):
        with open(data_path) as dataset_file:
            dataset = []
            for line in dataset_file:
                this_item = json.loads(line)
                if args.recomp_data == "hotpotqa" and this_item["f1"] <= 0.6:
                    continue
                dataset.append(this_item)
    else:
        with open(data_path) as dataset_file:
            dataset = json.load(dataset_file)
    
    recomp_items = []
    # levels = []
    for item in dataset:
        # obtain question
        question = item[decomp_dict[args.recomp_data]["question"]]
        context_list = ""
        # obtain decomposition
        if args.recomp_data == "musique":
            decomposition = item[decomp_dict[args.recomp_data]["decomp"]]
            temp = []
            for decomp in decomposition:
                temp.append(decomp["question"])
            decomposition = temp
        else:
            try:
                decomposition = item[decomp_dict[args.recomp_data]["decomp"]]
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
            if context:
                support_facts = item["supporting_facts"]
                all_paras = item["context"]
                context_list = []
                for fact in support_facts:
                    fact = fact[0]
                    for para in all_paras:
                        if fact == para[0]:
                            this_context = "".join(para[1])
                            context_list.append(this_context)
        elif args.recomp_data == "musique":
            sub_answer = item[decomp_dict[args.recomp_data]["decomp"]]
            temp = []
            for decomp in sub_answer:
                temp.append(decomp["answer"])
            sub_answer = temp
            label = item["answer"]
            if context:
                paras = item["paragraphs"]
                context_list = []
                for para in paras:
                    if para["is_supporting"]:
                        context_list.append(para["paragraph_text"])
        this_item = {
            "question": question,
            "decomposition": decomposition,
            "sub-answers": sub_answer,
            "label": label,
            "context": context_list
            }
        recomp_items.append(this_item)
        # levels.append(item["level"])
        
    return recomp_items

def load_bsldataset(args):
    data_file = decomp_dict[args.eval_data]["test"]
    data_path = f"{args.root_path}/data/{args.eval_data}/{data_file}"
    if args.eval_data == "musique":
        with open(data_path) as f:
            val_dataset = []
            for line in f:
                this_item = json.loads(line)
                val_dataset.append(this_item)
    else:
        with open(data_path) as f:
            val_dataset = json.load(f)
    output = []
    for sample in val_dataset:
        
        if args.eval_data == "strategyQA":
            item = {
                "question": sample["question"],
                "context": "",
                "label": 1 if sample["answer"] == True else 0
            }
        elif args.eval_data == "hotpotqa-yn":
            item = {
                "question": sample["question"],
                "context": "",
                "label": 1 if sample["answer"].lower() == "yes" else 0
            }
        elif args.eval_data == "hotpotqa":
            support_facts = sample["supporting_facts"]
            context_list = ""
            if args.use_context:
                all_paras = sample["context"]
                context_list = []
                for fact in support_facts:
                    fact = fact[0]
                    for para in all_paras:
                        if fact == para[0]:
                            this_context = "".join(para[1])
                            context_list.append(this_context)
            item = {
                "question": sample["question"],
                "context": context_list,
                "label": sample["answer"]
            }
        elif args.eval_data == "musique":
            context_list = ""
            if args.use_context:
                paras = sample["paragraphs"]
                context_list = []
                for para in paras:
                    if para["is_supporting"]:
                        context_list.append(para["paragraph_text"])
            item = {
                "question": sample["question"],
                "context": context_list,
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

def load_context(data_name, split="train", args=None):
    data_file = decomp_dict[data_name][split]
    data_path = f"{args.root_path}/data/{data_name}/{data_file}"
    if data_name=="musique" :
        with open(data_path) as dataset_file:
            dataset = []
            for line in dataset_file:
                this_item = json.loads(line)
                dataset.append(this_item)
    elif data_name=="hotpotqa":
        with open(data_path) as dataset_file:
            dataset = json.load(dataset_file)
    
    question2context = {}
    # levels = []
    for item in tqdm(dataset):
        # obtain question
        question = item[decomp_dict[data_name]["question"]]
        question = format_question(question)
        # quesiton = tokenizer.decode(tokenizer.encode(question), skip_special_tokens=True)
        if data_name == "hotpotqa":
            support_facts = item["supporting_facts"]
            all_paras = item["context"]
            context_list = []
            for fact in support_facts:
                fact = fact[0]
                for para in all_paras:
                    if fact == para[0]:
                        this_context = "".join(para[1])
                        context_list.append(this_context)
            question2context[question] = context_list
        elif data_name == "musique":
            paras = item["paragraphs"]
            context_list = []
            for para in paras:
                if para["is_supporting"]:
                    context_list.append(para["paragraph_text"])
            question2context[question] = context_list
        
    return question2context

def load_label(data_name, tokenizer, split="train", args=None):
    question2label = {}
    file_name = decomp_dict[data_name][split]
    data_path = f"{args.root_path}/data/{data_name}/{file_name}"
    # print(data_path)
    with open(data_path) as f:
        ref_dataset = json.load(f)
    # print(len(ref_dataset))
    # print(ref_dataset[:5])
    for item in tqdm(ref_dataset):
        question, label = item[decomp_dict[data_name]["question"]], item["answer"]
        question = format_question(question)
        question2label[question] = label
    return question2label

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
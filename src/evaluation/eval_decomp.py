from utils.param import parse_args
from utils.utils import *
from utils.globals import *
from utils.score_utils import *

from models.causal_llm import *
from models.gpt import *

from dataset.data import RecompDataset
from dataset.get_data import load_context

from torch.utils.data import DataLoader
from transformers import default_data_collator

import re
import json
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import numpy as np
import random

def create_prompt_qualify(question, decompositions, args):
    chat_template = chat_templates[args.eval_model]
    indexes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    template = qualify_template
    cnt = 0
    choices = ""
    for decomp in decompositions:
        this_idx = indexes[cnt]
        this_choice = this_idx + "."+ decomp + '\n'
        choices += this_choice
        cnt += 1
    prompt = template.format(question = question, choices = choices)
    prompt = chat_template.format(prompt=prompt)
    return prompt

def create_prompt_attribute(attribute, decomp_list, args):
    chat_template = chat_templates[args.eval_model]
    template = has_attr_template
    cnt = 0
    decomp_str = ""
    for decomp in decomp_list:
        cnt += 1
        decomp_str += f"{cnt}. {decomp}"
    
    prompt = template.format(attribute = attribute, sentences = decomp_str)
    prompt = chat_template.format(prompt=prompt)
    return prompt

def reformat_dataset(dataset, args):
    # construct the reference dataset
    question2label = {}
    file_name = decomp_dict[args.decomp_data]["test"]
    data_path = f"{args.root_path}/data/{args.decomp_data}/{file_name}"
    with open(data_path) as f:
        if args.decomp_data == "musique":
            with open(data_path) as f:
                ref_dataset = []
                for line in f:
                    this_item = json.loads(line)
                    ref_dataset.append(this_item)
        else:
            ref_dataset = json.load(f)
    for item in ref_dataset:
        if args.decomp_data == "strategyQA":
            label = item["answer"]
            label = 1 if label == True else 0
        elif args.decomp_data == "hotpotqa-yn":
            label = item["answer"]
            label = 1 if label.lower() == "yes" else 0
        elif args.decomp_data in ("hotpotqa", "musique"):
            label = item["answer"]
        question = format_question(item["question"])
        question2label[question] = label
    # reformat the original dataset
    output = []
    for sample in dataset:
        question = format_question(sample["question"])
        decompostions = sample["decomposition"]
        subanswers = sample["subanswer"]
        for decomp, subans in zip(decompostions, subanswers):
            if len(subans) > 0:
                decomp_list = decomp.split("<s>")
                decomp_list = [sent.strip() for sent in decomp_list]
                label = question2label[question]
                context = ""
                if args.use_context:
                    context = question2context[question]
                item = {
                    "question": question,
                    "decomposition": decomp_list,
                    "sub-answers": subans,
                    "context": context,
                    "label": label
                }
                output.append(item)
    return output

def get_eval_metric_cls(labels, predictions):
    lbl_list = []
    pred_list = []
    for key in labels:
        this_lbl = labels[key][0]
        this_preds = predictions[key]
        lbl_list.append(this_lbl)
        match_pred = this_preds[0]
        for pred in this_preds:
            if pred == this_lbl:
                match_pred = pred
        pred_list.append(match_pred)

        # this_lbl = labels[key][0]
        # this_preds = predictions[key]
        # for pred in this_preds:
        #     lbl_list.append(this_lbl)
        #     pred_list.append(pred)

    acc = accuracy_score(lbl_list, pred_list)
    auc = roc_auc_score(lbl_list, pred_list)
    return acc, auc

def get_eval_metric_qa(labels, predictions):
    lbl_list = []
    pred_list = []
    for key in labels:
        this_lbl = labels[key][0]
        this_preds = predictions[key]
        lbl_list.append(this_lbl)
        best_f1 = 0
        best_pred = this_preds[0]
        for pred in this_preds:
            f1 = f1_score([pred], [this_lbl])
            if f1 > best_f1:
                best_pred = pred
                best_f1 = f1
        pred_list.append(best_pred)

    rougnL = rouge(pred_list, lbl_list)
    f1 = f1_score(pred_list, lbl_list)
    exact_match = exact_match_score(pred_list, lbl_list)
    return rougnL, f1, exact_match

def create_attr_dict(dataset):
    attr_dict = {}
    for item in dataset:
        attr_dict[item["question"]] = item["private attributes"]
    return attr_dict

if __name__ == '__main__':
    args = parse_args()
    data_type = dataset_type[args.decomp_data]

    if args.use_context:
        question2context = load_context(args.decomp_data, split="test", args=args)

    if args.eval_qualify:
        model = get_eval_model(args, model_name = "gpt-4-turbo")
        model_name = args.decomp_model.split("/")[-1]
        data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}.json"
        dataset = read_data(data_path)[:(200*5)]
        dataset = sing2multi(["question"], ["decomposition"], dataset)
        output = []
        output_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}_eval.json"
        # extract the qualified decompositions
        with tqdm(total=len(dataset)) as pbar:
            for i in range(len(dataset)):
                item = dataset[i]
                question, decompositions = item["question"], item["decomposition"]
                prompt = create_prompt_qualify(question, decompositions, args)
                prompts = [prompt]
                # print(prompts)
                quallify_cands = model.generate(prompts)
                # print(quallify_cands)
                item["qualified decomp"] = quallify_cands[0]
                output.append(item)
                if (i+1) % 10 == 0 or i+1 == len(dataset):
                    write_list(output_path, output)
                pbar.update(1)
    
    if args.query_gpt:
        model = get_eval_model(args)
        # filter the qualified decomp
        model_name = args.decomp_model.split("/")[-1]
        if args.skip_eval_quality:
            data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}.json"
            dataset = read_data(data_path)
            dataset = sing2multi(["question"], ["decomposition"], dataset)
            random.shuffle(dataset)
            dataset = dataset[:200]
        else:
            data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}_eval.json"
            dataset = read_data(data_path)
            temp = []
            for item in dataset:
                quallify_cands = item["qualified decomp"]
                quallify_cands = process_candidate(quallify_cands)
                if len(quallify_cands) > 0:
                    decomps = item["decomposition"]
                    item["decomposition"] = [decomps[i] for i in quallify_cands]
                    temp.append(item)
            dataset = temp
        # print(dataset[:2])
        # extract answers from qualified decompositions
        output = []
        output_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}_eval_{args.eval_model}_v2.json"
        with tqdm(total=len(dataset)) as pbar:
            for item in dataset:
                decomps = item["decomposition"]
                item["subanswer"] = []
                question = item["question"]
                question = format_question(question)
                if args.use_context:
                    context_list = question2context[question]
                    context_pre = "Context: "
                    for context in context_list:
                        context_pre += context + " "
                for decomp in decomps:
                    subanswers = []
                    decomp_list = decomp.split("<s>")
                    pre_subq = ""
                    for cnt, subq in enumerate(decomp_list[:-1], start=1):
                        if args.abbrevation:
                            prompt = f"{pre_subq} Please answer the abbreviated question as short as possible: {subq}"
                        else:
                            prompt = subquery_template.format(question=f"{pre_subq}{subq}")
                        if args.use_context:
                            prompt = context_pre + prompt
                        # print(prompt)
                        subanswer = model.generate([prompt])
                        pre_subq += f"#{cnt}: {subanswer[0]} "
                        subanswers.append(subanswer[0])
                    item["subanswer"].append(subanswers)
                output.append(item)
                write_list(output_path, output)
                pbar.update(1)

    if args.eval_utility:
        model_name = args.decomp_model.split("/")[-1]
        # evaluate the final answer
        data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}_eval_{args.eval_model}_v2.json"
        dataset = read_data(data_path)
        # reformat the dataset
        dataset = reformat_dataset(dataset, args)
        model_name = args.recomp_model.split("/")[-1]
        if data_type == "qa":
            model_dir = f"{args.root_path}/save_models/recomp/{args.recomp_data}/{model_name}_{str(args.pretrain_recomp)}_{str(args.use_context)}"
        else:
            model_dir = f"{args.root_path}/save_models/recomp/{args.recomp_data}/{model_name}"
        if data_type == "qa":
            tokenizer, model = get_model_tokenizer_qa(model_dir)
            classification = False
        else:
            tokenizer, model = get_model_tokenizer_cls(model_dir, 2)
            classification = True
        causal_model = model_causal[args.recomp_model]
        dataset = RecompDataset(dataset, tokenizer, classification=classification, causal=causal_model, test=True)
        dataloader = DataLoader(
                dataset, 
                batch_size=args.test_batch_size, 
                collate_fn=default_data_collator, 
                pin_memory=True,
                shuffle=True,
                )
        
        # evaluate the utility
        labels = {}
        predictions = {}
        model.eval()
        with tqdm(total=len(dataloader)) as pbar:
            for i, batch in enumerate(dataloader):
                for key in batch:
                    batch[key] = batch[key].to(model.device)
                
                questions = tokenizer.batch_decode(batch["question"], skip_special_tokens=True)
                if data_type == "qa":
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
                    this_labels = []
                    for label_ids in batch["labels"]:
                        label_ids = label_ids[label_ids != IGNORE_INDEX]
                        label = tokenizer.decode(label_ids, skip_special_tokens=True)
                        this_labels.append(label)
                else:
                    with torch.no_grad():
                        outputs = model(
                                input_ids = batch["input_ids"], 
                                attention_mask = batch["attention_mask"])
                    logits = outputs.logits
                    y_pred = torch.argmax(logits, -1)
                    y_preds = y_pred.tolist()
                    this_labels = batch["labels"].tolist()

                for question, pred, label in zip(questions, y_preds, this_labels):
                    if question in labels.keys():
                        labels[question].append(label)
                        predictions[question].append(pred)
                    else:
                        labels[question] = [label]
                        predictions[question] = [pred]
                pbar.update(1)
                if data_type == "qa":
                    rougnL, f1, exact_match = get_eval_metric_qa(labels, predictions)
                    pbar.set_postfix(rougnL=rougnL, f1=f1, exact_match=exact_match)
                else:
                    acc, auc = get_eval_metric_cls(labels, predictions)
                    pbar.set_postfix(acc=acc, auc=auc)
            if data_type == "qa":
                print(f"RougnL: {rougnL}")
                print(f"F1: {f1}")
                print(f"Exact match: {exact_match}")
            else:
                print(f"Accuracy: {acc}")
                print(f"AUC: {auc}")
    
    # evaluate the complexity
    # judge whether each decomposition contains the attribute
    if args.eval_complexity:
        if args.extract_attr:
            model = get_eval_model(args, model_name = "gpt-4-turbo")
            model_name = args.decomp_model.split("/")[-1]
            data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}_eval_{args.eval_model}_v2.json"
            output_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}_eval_{args.eval_model}_v3.json"
            dataset = read_data(data_path)
            dataset = extract_attribute(dataset, args)
            outputs = []
            # outputs = read_data(output_path)
            with tqdm(total=len(dataset)) as pbar:
                for i, item in enumerate(dataset):
                    if i < len(outputs):
                        pbar.update(1)
                        continue
                    attrs, decompositions = item["attributes"], item["decomposition"]
                    decompositions = decompositions[:-1]
                    attrs2decomp = {}
                    for attr in attrs:
                        prompt = create_prompt_attribute(attr, decompositions, args)
                        prompts = [prompt]
                        sents = model.generate(prompts)
                        attrs2decomp[attr] = sents[0]
                    item["has_attrs"] = attrs2decomp
                    outputs.append(item)
                    write_list(output_path, outputs)
                    pbar.update(1)
        
        # compute complexity based on the number of attributes
        model_name = args.decomp_model.split("/")[-1]
        data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}/decompose_{args.test_mode}_test_epoch_{args.test_epoch}_eval_{args.eval_model}_v3.json"
        dataset = read_data(data_path)
        # compute the average token length for the question
        question_tokens = []
        for item in dataset:
            question = item["question"]
            query_tok = num_tokens_from_string(question)
            question_tokens.append(query_tok)
        avg_query_tok = np.mean(question_tokens)
        print(f"Average token length for the raw question: {avg_query_tok}")

        for mu in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            args.mu = 1/mu
            complexities = {}
            query_costs = {}
            
            for item in dataset:
                question, attrs, attrs2decomp, decompositions, subanswer = item["question"], item["attributes"], item["has_attrs"], item["decomposition"], item["subanswer"]
                
                # convert decompositons into list of sub-questions
                decomp_list = decompositions[:-1]
                # create a dictionary for decomposition to attribute
                decomp2attrs = {idx:[] for idx in range(len(decomp_list))} # {index of decomp: [attributes it contains]}
                for attr in attrs2decomp:
                    decomp_string = attrs2decomp[attr]
                    decomp_idx = process_has_attr(decomp_string)
                    for idx in decomp_idx:
                        if idx <= len(decomp_list):
                            decomp2attrs[idx-1].append(attr)
                # compute the complexity for each query
                complexity = 0
                query_cost = 0
                for i, (single_query, signle_ans) in enumerate(zip(decomp_list, subanswer)):
                    # if the query contain answers from previous question like #1 or #2, 
                    # find the related attributes in previous question, and add the attribute
                    indexes = re.findall(r'#(\d+)', single_query)
                    for idx in indexes:
                        idx = int(idx)
                        # ensure the validity of the index
                        if idx <= len(decomp_list):
                            related_query_attrs = decomp2attrs[idx-1]
                            decomp2attrs[i] = decomp2attrs[i] + related_query_attrs
                    attr_cnt = len(set(decomp2attrs[i]))
                    if attr_cnt > 0:
                        n_input_toks = num_tokens_from_string(single_query)
                        n_output_toks =num_tokens_from_string(signle_ans)
                        token_cost = n_input_toks * token_cost_dict[args.eval_model][0] + n_output_toks * token_cost_dict[args.eval_model][1]
                        this_complexity = (1/args.mu) ** attr_cnt
                        complexity += this_complexity
                        query_cost += complexity * token_cost
                        
                if question in complexities.keys():
                    pre_complex = complexities[question]
                    complexities[question] = min(complexity, pre_complex)
                    pre_cost = query_costs[question]
                    query_costs[question] = min(query_cost, pre_cost)
                else:
                    complexities[question] = complexity
                    query_costs[question] = query_cost
            avg_complex = np.mean(list(complexities.values()))
            avg_query_cost = np.mean(list(query_costs.values()))
            print(f"Average complexity for 1/mu={mu}: {avg_complex}")
            print(f"Average query cost for 1/mu={mu}: {avg_query_cost}")

        for mu in [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            if mu == 0:
                args.mu =0
            else:
                args.mu = 1/mu
            # compute the complexity without decompostion
            data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/test_decompose_all.json"
            dataset = read_data(data_path)
            query2attr = create_attr_dict(dataset)
            for suffix in ["cot", "dq"]:
                data_path = f"{args.root_path}/results/{args.decomp_data}/baseline/predictions_{args.eval_model}_{suffix}.json"
                dataset = read_data(data_path)
                complexities = []
                query_costs = []
                for item in dataset:
                    question, answer = item["question"], item["raw_answer"]
                    attributes = query2attr[question]
                    n_attr = len(attributes)
                    if args.mu == 0:
                        complexity = 1
                    else:
                        complexity = (1/args.mu) ** n_attr
                    complexities.append(complexity)
                    n_input_toks = num_tokens_from_string(question)
                    n_output_toks =num_tokens_from_string(answer)
                    token_cost = n_input_toks * token_cost_dict[args.eval_model][0] + n_output_toks * token_cost_dict[args.eval_model][1]
                    query_costs.append(token_cost * complexity)
                    
                avg_complex = np.mean(complexities)
                avg_query_cost = np.mean(query_costs)
                print(f"Average complexity for 1/mu={mu} without decompositions in case {suffix}: {avg_complex}")
                print(f"Average query cost for 1/mu={mu} without decompositions in case {suffix}: {avg_query_cost}")
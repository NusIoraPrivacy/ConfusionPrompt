from utils.param import parse_args
from utils.utils import *
from utils.score_utils import *
from utils.globals import *
from dataset.get_data import *

from tqdm import tqdm
import numpy as np


def create_prompt_decomp(args, question, answer, attribute=None):
    if args.use_attr:
        prompt = decomp_template_attr.format(question=question, attributes=attribute, answer=answer)
    else:
        prompt = decomp_template.format(question=question, answer=answer)
    return prompt

def create_prompt_attr(args, question):
    prompt = find_attr_template.format(sentence=question)
    return prompt

def get_avg_utility(dataset):
    f1 = []
    roughL = []
    exact_match = []
    for sample in dataset:
        f1.append(sample["f1"])
        roughL.append(sample["rougeL"])
        exact_match.append(sample["exact_match"])
    f1 = sum(f1)/len(f1)
    roughL = sum(roughL)/len(roughL)
    exact_match = sum(exact_match)/len(exact_match)
    print(f"Average f1 score: {f1}, average roughL: {roughL}, average exact match: {exact_match}")

if __name__ == '__main__':
    args = parse_args()
    model = get_eval_model(args)
    dataset = load_recomp_dataset(args, split=args.mode)[:2000]

    if args.extract_attr:
        output_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs.json"
        output = []
        cnt = 0
        for sample in tqdm(dataset):
            question = sample["question"]
            prompt = create_prompt_attr(args, question)
            n_tries = 0
            while True:
                attrs = model.generate([prompt])
                try:
                    attrs = eval(attrs[0])
                    break
                except Exception as e:
                    n_tries += 1
                if n_tries > 5:
                    attrs = []
                    break
            this_item = {"question": question, "private attributes": attrs, "label": sample["label"]}
            output.append(this_item)
            cnt += 1
            if cnt % 5 == 0:
                write_list(output_path, output)
    
    if args.extract_decomp:
        data_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs.json"
        dataset = read_data(data_path)
        output_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs_decomp_{args.eval_model}.json"
        output = []
        cnt = 0
        for sample in tqdm(dataset):
            question, answer, attributes = sample["question"], sample["label"], sample["private attributes"]
            prompt = create_prompt_decomp(args, question, answer, attributes)
            n_tries = 0
            while True:
                decomps = model.generate([prompt])
                try:
                    decomps = eval(decomps[0])
                    break
                except Exception as e:
                    n_tries += 1
                if n_tries > 5:
                    decomps = []
                    break
            sample["decomposition"] = decomps
            output.append(sample)
            cnt += 1
            if cnt % 5 == 0:
                write_list(output_path, output)
    
    if args.val:
        data_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs_decomp_{args.eval_model}.json"
        dataset = read_data(data_path)
        output_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs_decomp_{args.eval_model}_ans.json"
        cnt = 0
        output = read_data(output_path)
        pre_questions = [item["question"] for item in output]
        for sample in tqdm(dataset):
            question, answer, decomp_list = sample["question"], sample["label"], sample["decomposition"]
            if question in pre_questions:
                continue
            subanswers = []
            pre_subq = ""
            for cnt, subq in enumerate(decomp_list, start=1):
                prompt = subquery_template.format(question=f"{pre_subq}{subq}")
                subanswer = model.generate([prompt])
                pre_subq += f"#{cnt}: {subanswer[0]} "
                subanswers.append(subanswer[0])
            if len(subanswers) > 0:
                prediction = subanswers[-1]
            else:
                prediction = ""
            sample["subanswers"] = subanswers
            sample["prediction"] = prediction
            # compute the score for the prediction
            sample["rougeL"] = rouge([prediction], [answer])
            sample["f1"] = f1_score([prediction], [answer])
            sample["exact_match"] = exact_match_score([prediction], [answer])
            
            output.append(sample)
            cnt += 1
            if cnt % 5 == 0:
                write_list(output_path, output)
    
    if args.redone_decomp:
        data_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs_decomp_{args.eval_model}_ans.json"
        output_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs_decomp_{args.eval_model}_ans_final.json"
        dataset = read_data(data_path)
        # compute the average utility before redone the decompostition extract
        get_avg_utility(dataset)
        cnt = 0
        outputs = read_data(output_path)
        question_list = [sample["question"] for sample in outputs]
        for sample in tqdm(dataset):
            if sample["question"] in question_list:
                continue
            if sample["f1"] <= 0.5:
                success = False
                n_tries = 0
                while not success:
                    question, answer, attributes = sample["question"], sample["label"], sample["private attributes"]
                    prompt = create_prompt_decomp(args, question, answer, attributes)
                    decomps = model.generate([prompt])
                    try:
                        decomps = eval(decomps[0])
                    except Exception as e:
                        decomps = []

                    subanswers = []
                    pre_subq = ""
                    for cnt, subq in enumerate(decomps, start=1):
                        prompt = subquery_template.format(question=f"{pre_subq}{subq}")
                        subanswer = model.generate([prompt])
                        pre_subq += f"#{cnt}: {subanswer[0]} "
                        subanswers.append(subanswer[0])
                    if len(subanswers) > 0:
                        prediction = subanswers[-1]
                    else:
                        prediction = ""
                    # compute the score for the prediction
                    f1 = f1_score([prediction], [answer])
                    if f1 > 0.5:
                        success = True
                        sample["decomposition"] = decomps
                        sample["subanswers"] = subanswers
                        sample["prediction"] = prediction
                        sample["rougeL"] = rouge([prediction], [answer])
                        sample["f1"] = f1
                        sample["exact_match"] = exact_match_score([prediction], [answer])
                    n_tries += 1
                    if n_tries > 5:
                        break
            outputs.append(sample)
            cnt += 1
            if cnt % 5 == 0:
                write_list(output_path, outputs)
            if cnt >= 200:
                break
        # compute the average utility after redone the decompostition extract
        get_avg_utility(outputs)

# data_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs_decomp_{args.eval_model}_ans_final.json"
# dataset = read_data(data_path)
# output_path = f"{args.root_path}/data/{args.recomp_data}/{args.mode}_attrs_decomp_{args.eval_model}_ans_final.json"
# cnt = 0
# output = []

# for sample in tqdm(dataset):
#     question, answer, decomp_list = sample["question"], sample["label"], sample["decomposition"]
#     if sample["f1"] > 0.6:
#         subanswers = []
#         pre_subq = ""
#         for cnt, subq in enumerate(decomp_list, start=1):
#             prompt = subquery_template.format(question=f"{pre_subq}{subq}")
#             subanswer = model.generate([prompt])
#             pre_subq += f"#{cnt}: {subanswer[0]} "
#             subanswers.append(subanswer[0])
#         sample["subanswers"] = subanswers
    
#     output.append(sample)
#     cnt += 1
#     if cnt % 5 == 0:
#         write_list(output_path, output)
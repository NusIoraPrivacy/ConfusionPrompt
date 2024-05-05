from utils.globals import *
from utils.utils import read_data, write_list, get_eval_model
from utils.param import str2bool
from tqdm import tqdm
import itertools
import argparse
import json
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))
GEN_FT_SAMPLE = True
GEN_REPLACE = True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--gen_ft_sample", type=str2bool, default=True)
    parser.add_argument( "--mutliple_resp", type=str2bool, default=True)
    parser.add_argument( "--gen_replace", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--n_replaces", type=int, default=10,
        help = "number of replacements for each question")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.gen_ft_sample:
        if args.mutliple_resp:
            model = get_eval_model(args, model_name = "gpt-4-turbo")
            data_path = f"{_ROOT_PATH}/results/strategyQA/replace/question_attrs.json"
            with open(data_path)  as f:
                dataset = json.load(f)
            ft_dataset = []
            cnt = 0
            for question, attrs in tqdm(dataset.items()):
                attr_combs = []
                for r in range(1, len(attrs) + 1):
                    attr_combs.extend(list(itertools.combinations(attrs, r)))
                for attr_comb in attr_combs:
                    attr_comb = list(attr_comb)
                    prompt = replace_template_multiple_resp.format(attributes=attr_comb, sentence=question, n_replaces=100)
                    success = False
                    n_try = 0
                    while not success:
                        replace_sents = model.generate(prompt)[0]
                        try:
                            replace_sents = eval(replace_sents)
                            success = True
                        except Exception as e:
                            n_try += 1
                        if n_try >= 5:
                            break
                    if success:
                        message = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(replace_sents)}]}
                        ft_dataset.append(message)
                        file_path = f"{_ROOT_PATH}/data/strategyQA/replace_gptft_samples_multiple.jsonl"
                        write_list(file_path, ft_dataset)
                cnt += 1
                if cnt >= 20:
                    break

        else:
            replace_path = f"{_ROOT_PATH}/data/strategyQA/replace_train_samples.json"

            dataset = read_data(replace_path)

            ft_dataset = []
            for item in dataset:
                raw_q, attr, replace_q = item["raw query"], item["attributes"], item["replaced query"]
                prompt = replace_template.format(attributes=attr, sentence=raw_q)
                message = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": replace_q}]}
                ft_dataset.append(message)
            
            file_path = f"{_ROOT_PATH}/data/strategyQA/replace_gptft_samples.jsonl"
            write_list(file_path, ft_dataset)

    if args.gen_replace:
        data_path = f"{_ROOT_PATH}/results/strategyQA/replace/question_attrs.json"
        with open(data_path)  as f:
            dataset = json.load(f)
        model = get_eval_model(args, model_name = "ft:gpt-3.5-turbo-1106:personal::9L8ch9My")

        out_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_candidates.json"
        outputs = []
        for question, attrs in tqdm(dataset.items()):
            attr_combs = []
            for r in range(1, len(attrs) + 1):
                attr_combs.extend(list(itertools.combinations(attrs, r)))
            for attr_comb in attr_combs:
                attr_comb = list(attr_comb)
                if args.mutliple_resp:
                    prompt = replace_template_multiple_resp.format(attributes=attr_comb, sentence=question, n_replaces=args.n_replaces)
                    success = False
                    n_try = 0
                    while not success:
                        replace_sents = model.generate(prompt)[0]
                        try:
                            replace_sents = eval(replace_sents)
                            success = True
                        except Exception as e:
                            n_try += 1
                        if n_try >= 5:
                            replace_sents = []
                            print(replace_sents)
                            break
                else:
                    prompt = replace_template.format(attributes=attr_comb, sentence=question)
                    replace_sents = []
                    for i in range(args.n_replaces):
                        replace_sent = model.generate(prompt)[0]
                        replace_sents.append(replace_sent)
                item = {"question": question, "attributes": attr_comb, "replace sentences": replace_sents}
                outputs.append(item)
                with open(out_path, "w") as f:
                    json.dump(outputs, out_path, indent=4)
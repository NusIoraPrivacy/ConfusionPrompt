from datasets import load_dataset
import json
import pandas as pd
from utils.globals import mmlu_decomp_template, mmlu_subanswer_template
from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from models.key import _API_KEY
import argparse
import os
import time
from tqdm import tqdm
import random
current = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(current))

cat_dict = {"business": ["business_ethics", "marketing"],
            "legal": ["international_law", "jurisprudence"],
            "politics": ["us_foreign_policy", "high_school_government_and_politics"],
            "medicine": ["college_medicine", "clinical_knowledge", "nutrition"],
            "religion": ["world_religions"]
            }

def creat_mmlu_dict(data_subset, cat):
    out = []
    questions = data_subset["question"]
    choices = data_subset["choices"]
    answers = data_subset["answer"]
    for question, choice, answer in zip(questions, choices, answers):
        this_dict = {
            "question": question,
            "choices": choice,
            "answer": answer,
            "category": cat,
        }
        out.append(this_dict)

    return out

def creat_mmlu_cat_data(root):
    filter_data = []
    for cat in cat_dict:
        sub_cats = cat_dict[cat]
        for sub_cat in sub_cats:
            data_subset = load_dataset("cais/mmlu", sub_cat, split="test")
            filter_data.extend(creat_mmlu_dict(data_subset, cat))
            data_subset = load_dataset("cais/mmlu", sub_cat, split="validation")
            filter_data.extend(creat_mmlu_dict(data_subset, cat))

    with open(f'{root}/data/mmlu/raw.json', 'w') as fout:
        json.dump(filter_data, fout, indent=4)


    df = pd.DataFrame.from_records(filter_data)
    df.to_csv(f"{root}/data/mmlu/raw.csv", index=False)

def create_decomp_prompt_template(demos, category, template, target_question, target_answer):
    related_data = []
    for sample in demos:
        if sample["category"] == category:
            related_data.append(sample)
    format_strings = []
    for sample in related_data:
        format_strings.append(sample["question"])
        answer = sample["choices"][sample["answer"]]
        format_strings.append(answer)
        decomp_list = sample["decomposition"]
        decomp_str = "["
        for i, decomp in enumerate(decomp_list):
            if i > 0:
                decomp_str += " "
            decomp_str += ('"' + decomp + '",')
        decomp_str += "]"
        format_strings.append(decomp_str)
    format_strings.extend([target_question, target_answer])
    template = template.format(*format_strings)
    return template

def create_subans_prompt_template(demos, category, template, target_question, target_answer, target_decomp_list):
    related_data = []
    for sample in demos:
        if sample["category"] == category:
            related_data.append(sample)
    format_strings = []
    for sample in related_data:
        question = sample["question"]
        answer = sample["choices"][sample["answer"]]
        decomp_list = sample["decomposition"]
        subans_list = sample["facts"]
        idx = random.choice([i for i in range(len(decomp_list))])
        format_strings.append(question)
        format_strings.append(answer)
        format_strings.append(decomp_list[idx])
        format_strings.append(subans_list[idx])
    
    templates = []
    for target_decomp in target_decomp_list:
        this_format_strings = format_strings + [target_question, target_answer, target_decomp]
        this_template = template.format(*this_format_strings)
        templates.append(this_template)
    return templates

def gen_decomp_samples_gpt(root):
    with open(f"{root}/data/mmlu/raw_decom.json") as f:
        decomp_demo = json.load(f)
    with open(f"{root}/data/mmlu/raw.json") as f:
        raw_data = json.load(f)
    prompts = []
    for sample in raw_data:
        cat = sample["category"]
        question = sample["question"]
        answer = sample["choices"][sample["answer"]]
        prompt = create_decomp_prompt_template(decomp_demo, cat, mmlu_decomp_template, question, answer)
        prompts.append(prompt)
    return prompts, raw_data

def gen_subans_samples_gpt(root):
    with open(f"{root}/data/mmlu/raw_decom.json") as f:
        decomp_demo = json.load(f)
    with open(f"{root}/data/mmlu/decomp_data.json") as f:
        raw_data = json.load(f)
    prompts = []
    for sample in raw_data:
        cat = sample["category"]
        question = sample["question"]
        answer = sample["choices"][sample["answer"]]
        decomp_list = sample["decomposition"]
        prompt_list = create_subans_prompt_template(decomp_demo, cat, mmlu_subanswer_template, question, answer, decomp_list)
        prompts.append(prompt_list)
    return prompts, raw_data

def create_message(prompt):
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages

def get_response(client, prompt, args):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    messages = create_message(prompt)
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            response = client.chat.completions.create(
                model=args.gpt_model,
                messages=messages,
                temperature=args.temperature,
                # max_tokens=args.max_tokens,
            )
            rslt = response.choices[0].message.content
            success = True
        except RateLimitError as e:
            print(f"sleep {SLEEP_TIME} seconds for rate limit error")
            time.sleep(SLEEP_TIME)
        except APITimeoutError as e:
            print(f"sleep {SLEEP_TIME} seconds for time out error")
            time.sleep(SLEEP_TIME)
        except APIConnectionError as e:
            print(f"sleep {SLEEP_TIME} seconds for api connection error")
            time.sleep(SLEEP_TIME)
        except APIError as e:
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(e)
            success = True
            rslt = "Error"
        cnt += 1
    return rslt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=root_path)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--gpt_model", type=str, default="gpt-4o")
    parser.add_argument("--decomp", type=bool, default=False)
    parser.add_argument("--subans", type=bool, default=True)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.decomp:
        prompts, raw_data = gen_decomp_samples_gpt(args.root_path)
        outputs = []
        client = OpenAI(api_key=_API_KEY)
        for prompt, sample in tqdm(zip(prompts, raw_data)):
            decomp = get_response(client, prompt, args)
            success = False
            n_try = 0
            while not success:
                try:
                    decomp = eval(decomp)
                    success = True
                except Exception as e:
                    n_try += 1
                if n_try >= 5:
                    break
            if success:
                sample["decomposition"] = decomp
                outputs.append(sample)
                with open(f'{args.root_path}/data/mmlu/decomp_data.json', 'w') as fout:
                    json.dump(outputs, fout, indent=4)

    if args.subans:
        prompts, raw_data = gen_subans_samples_gpt(args.root_path)
        # print(prompts[0][1])
        outputs = []
        client = OpenAI(api_key=_API_KEY)
        for prompt_list, sample in tqdm(zip(prompts, raw_data)):
            all_subans = []
            for prompt in prompt_list:
                subans = get_response(client, prompt, args)
                all_subans.append(subans)
            sample["facts"] = all_subans
            outputs.append(sample)
            with open(f'{args.root_path}/data/mmlu/final_data.json', 'w') as fout:
                json.dump(outputs, fout, indent=4)
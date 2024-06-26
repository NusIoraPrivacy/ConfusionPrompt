from utils.globals import *
from utils.utils import read_data, write_list, get_eval_model
from utils.param import str2bool

from tqdm import tqdm
import itertools
import argparse
import json
import os
import time
from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from models.key import _API_KEY

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))
GEN_FT_SAMPLE = True
GEN_REPLACE = True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--gen_ft_sample", type=str2bool, default=False)
    parser.add_argument( "--gen_ft_sample_phrase", type=str2bool, default=False)
    parser.add_argument( "--mutliple_resp", type=str2bool, default=False)
    parser.add_argument( "--gen_replace", type=str2bool, default=False)
    parser.add_argument( "--gen_replace_phrase", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--n_replaces", type=int, default=10,
        help = "number of replacements for each question")
    parser.add_argument("--eval_model", type=str, default="gpt-4-turbo")
    parser.add_argument("--extract_phrase", type=str2bool, default=False)
    parser.add_argument("--extract_phrase_ft", type=str2bool, default=False)
    parser.add_argument("--gen_replace_local", type=str2bool, default=True)
    
    args = parser.parse_args()

    return args

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
                model=args.eval_model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
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

if __name__ == "__main__":
    args = parse_args()
    if args.gen_ft_sample:
        if args.mutliple_resp:
            client = OpenAI(api_key=_API_KEY)
            data_path = f"{_ROOT_PATH}/results/strategyQA/replace/question_attrs.json"
            with open(data_path)  as f:
                dataset = json.load(f)
            temp = {}
            cnt = 0
            for key in dataset:
                temp[key] = dataset[key]
                cnt += 1
                if cnt >= 50:
                    break
            dataset = temp
            ft_dataset = []

            for question, attrs in tqdm(dataset.items()):
                attr_combs = []
                for r in range(1, len(attrs) + 1):
                    attr_combs.extend(list(itertools.combinations(attrs, r)))
                for attr_comb in attr_combs:
                    attr_comb = list(attr_comb)
                    prompt = replace_template_multiple_resp.format(attributes=attr_comb, sentence=question, n_replaces=args.n_replaces)
                    success = False
                    n_try = 0
                    while not success:
                        replace_sents = get_response(client, prompt, args)
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
    
    if args.gen_ft_sample_phrase:
        if args.mutliple_resp:
            client = OpenAI(api_key=_API_KEY)
            data_path = f"{_ROOT_PATH}/results/strategyQA/replace/question_attrs.json"
            with open(data_path)  as f:
                dataset = json.load(f)
            temp = {}
            cnt = 0
            for key in dataset:
                temp[key] = dataset[key]
                cnt += 1
                if cnt >= 50:
                    break
            dataset = temp
            ft_dataset = []

            for question, attrs in tqdm(dataset.items()):
                attr_combs = []
                for r in range(1, len(attrs) + 1):
                    attr_combs.extend(list(itertools.combinations(attrs, r)))
                for attr_comb in attr_combs:
                    attr_comb = list(attr_comb)
                    for n_rpl in [5, 10]:
                        prompt = replace_template_phrase_multiple.format(attributes=attr_comb, sentence=question, n_replaces=n_rpl)
                        success = False
                        n_try = 0
                        while not success:
                            replace_phrases = get_response(client, prompt, args)
                            try:
                                replace_phrases = eval(replace_phrases)
                                if len(replace_phrases) == n_rpl and len(replace_phrases[0])==len(attr_comb):
                                    success = True
                                else:
                                    n_try += 1
                            except Exception as e:
                                n_try += 1
                            if n_try >= 5:
                                break
                        if success:
                            message = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(replace_phrases)}]}
                            ft_dataset.append(message)
                            file_path = f"{_ROOT_PATH}/data/strategyQA/replace_gptft_phrases_multiple.jsonl"
                            write_list(file_path, ft_dataset)

    if args.gen_replace:
        data_path = f"{_ROOT_PATH}/results/strategyQA/replace/question_attrs.json"
        with open(data_path)  as f:
            dataset = json.load(f)
        client = OpenAI(api_key=_API_KEY)
        args.eval_model = "ft:gpt-3.5-turbo-1106:personal:multiple-replace:9LOiijuO"

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
                    n_success = 0
                    n_try = 0
                    replace_sents = []
                    while n_success <= 3:
                        replace_sents = get_response(client, prompt, args)
                        try:
                            replace_sents += eval(replace_sents)
                            n_success += 1
                        except Exception as e:
                            n_try += 1
                        if n_try >= 15:
                            break
                else:
                    prompt = replace_template.format(attributes=attr_comb, sentence=question)
                    replace_sents = []
                    for i in range(args.n_replaces):
                        replace_sents = get_response(client, prompt, args)
                        replace_sents.append(replace_sent)
                item = {"question": question, "attributes": attr_comb, "replace sentences": replace_sents}
                outputs.append(item)
                with open(out_path, "w") as f:
                    json.dump(outputs, f, indent=4)
    
    if args.gen_replace_phrase:
        data_path = f"{_ROOT_PATH}/results/strategyQA/replace/question_attrs.json"
        with open(data_path)  as f:
            dataset = json.load(f)
        client = OpenAI(api_key=_API_KEY)
        args.eval_model = "ft:gpt-3.5-turbo-1106:personal:multiple-replace:9LOiijuO"

        out_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_phrase_candidates.json"
        outputs = []
        for question, attrs in tqdm(dataset.items()):
            attr_combs = []
            for r in range(1, len(attrs) + 1):
                attr_combs.extend(list(itertools.combinations(attrs, r)))
            for attr_comb in attr_combs:
                attr_comb = list(attr_comb)
                if args.mutliple_resp:
                    prompt = replace_template_phrase_multiple.format(attributes=attr_comb, sentence=question, n_replaces=args.n_replaces)
                    success = False
                    n_try = 0
                    while not success:
                        replace_phrases = get_response(client, prompt, args)
                        try:
                            replace_phrases = eval(replace_phrases)
                            if len(replace_phrases) == args.n_replaces and len(replace_phrases[0])==len(attr_comb):
                                success = True
                            else:
                                n_try += 1
                        except Exception as e:
                            n_try += 1
                        if n_try >= 5:
                            break
                    item = {"question": question, "attributes": attr_comb, "replace attributes": replace_phrases}
                    outputs.append(item)
                    with open(out_path, "w") as f:
                        json.dump(outputs, f, indent=4)

    if args.gen_replace_local:
        from dataset.data import ReplaceDataset
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, default_data_collator
        from torch.utils.data import DataLoader
        import random
        from utils.utils import extract_attr_query, sing2multi, format_question

        data_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_candidates.json"
        with open(data_path) as f:
            dataset = json.load(f)
        replace_inputs = []
        for sample in dataset:
            question, attributes, replacements = sample["question"], sample["attributes"], sample["replace sentences"]
            this_item = {"raw query": question, "attributes": attributes, "replaced query": ""}
            replace_inputs.append(this_item)
        # print(len(replace_inputs))

        model_dir = f"{_ROOT_PATH}/save_models/replace/strategyQA/sp/bart-large/final"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="cuda:1")
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        rpl_dataset = ReplaceDataset(replace_inputs, tokenizer)
        test_dataloader = DataLoader(
            rpl_dataset, 
            batch_size=10, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=1,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,
        )
        # generate decomposed questions with trained model
        final_outputs = []
        
        with tqdm(total=len(test_dataloader), unit='batch') as pbar:

            for step, batch in enumerate(test_dataloader):
                # just query gpt when epoch == 0
                for key in batch.keys():
                    batch[key] = batch[key].to(model.device)
                output_list = []
                for i in range(150):
                    output_ids = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            # **batch,
                            generation_config = generation_config,
                        )
                    responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                    questions = tokenizer.batch_decode(batch["input_ids"])
                    attr_query_lst = extract_attr_query(questions, tokenizer)
                    for item, resp in zip(attr_query_lst, responses):
                        item["replacement"] = resp
                        output_list.append(item)
                output_list = sorted(output_list, key=lambda d: (d["raw query"], d["attributes"])) 
                output_list = sing2multi(["raw query", "attributes"], ["replacement"], output_list)
                for item in output_list:
                    question, attributes, replacements = item["raw query"], item["attributes"], item["replacement"]
                    # found_match = False
                    # for sample in dataset:
                    #     question, attributes = sample["question"], sample["attributes"]
                    #     question = format_question(question)
                        # if question==this_query and set(attributes) == set(this_attr):
                        #     replacements = sample["replace sentences"]
                        #     try:
                        #         replacements = eval(replacements)
                        #     except Exception as e:
                        #         break
                        #     random.shuffle(replacements)
                        #     replacements = item["replacement"] + replacements[:-10]
                        #     this_out = {"question": question, "attributes": attributes, "replace sentences": replacements}
                        #     final_outputs.append(this_out)
                        #     found_match = True
                        #     break
                    this_out = {"question": question, "attributes": attributes, "replace sentences": replacements}
                    final_outputs.append(this_out)

                # store the file
                output_dir = f"{_ROOT_PATH}/results/strategyQA/replace"
                output_path = f"{output_dir}/replace_candidates_local.json"
                write_list(output_path, final_outputs)
                pbar.update(1)
    
    if args.extract_phrase_ft:
        data_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_candidates.json"
        with open(data_path) as f:
            dataset = json.load(f)
        dataset = dataset[:50]
        ft_dataset = []
        client = OpenAI(api_key=_API_KEY)
        for item in tqdm(dataset):
            question, attributes, rpl_sents = item["question"], item["attributes"], item["replace sentences"]
            try:
                rpl_sents = eval(rpl_sents)
            except Exception as e:
                continue
            
            prompt = extract_multiple_phrase_template.format(raw_query=question, attribute=attributes, replace_query=rpl_sents[0])
            # print(prompt)
            success = False
            n_try = 0
            while not success:
                alt_attrs = get_response(client, prompt, args)
                # print(alt_attrs)
                try:
                    alt_attrs = eval(alt_attrs)
                    success = True
                except Exception as e:
                    n_try += 1
                if n_try >= 5:
                    break
            
            if success:
                message = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(alt_attrs)}]}
                ft_dataset.append(message)
                file_path = f"{_ROOT_PATH}/data/strategyQA/replace_gptft_extract_attr.jsonl"
                write_list(file_path, ft_dataset)
    
    if args.extract_phrase:
        data_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_candidates.json"
        with open(data_path) as f:
            dataset = json.load(f)
        client = OpenAI(api_key=_API_KEY)
        args.eval_model = "ft:gpt-3.5-turbo-1106:personal:extract-attr:9Lkwmjux"

        out_path = f"{_ROOT_PATH}/results/strategyQA/replace/replace_attributes.json"
        outputs = []
        for cnt in range(3):
            for item in tqdm(dataset):
                question, attributes, rpl_sents = item["question"], item["attributes"], item["replace sentences"]
                try:
                    rpl_sents = eval(rpl_sents)
                except Exception as e:
                    continue
                
                prompt = extract_multiple_phrase_template.format(raw_query=question, attribute=attributes, replace_query=rpl_sents[cnt])
                # print(prompt)
                success = False
                n_try = 0
                while not success:
                    alt_attrs = get_response(client, prompt, args)
                    # print(alt_attrs)
                    try:
                        alt_attrs = eval(alt_attrs)
                        success = True
                    except Exception as e:
                        n_try += 1
                    if n_try >= 5:
                        break
                
                if success:
                    item = {"original question": question, "replaced question": rpl_sents[cnt], "attributes": attributes, "replace attributes": alt_attrs}
                    outputs.append(item)
                    with open(out_path, "w") as f:
                        json.dump(outputs, f, indent=4)
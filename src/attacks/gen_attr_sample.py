from utils.globals import *
from utils.param import str2bool
from utils.utils import get_eval_model, write_list, get_model_tokenizer_qa

from dataset.data import TweetRplDataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from transformers import default_data_collator, get_scheduler

import argparse
import json
import os
import pandas as pd
import pickle
from tqdm import tqdm

from openai import OpenAI
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)
from models.key import _API_KEY
import time

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="facebook/bart-large")
    parser.add_argument("--para_model", type=str, default="eugenesiow/bart-paraphrase",
                        choices=["google/flan-t5-xl", "eugenesiow/bart-paraphrase"])
    parser.add_argument("--t2t_model", type=str, default="facebook/bart-large")
    parser.add_argument("--generator", type=str, default="facebook/bart-large")
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--eval_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--max_new_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--lr", type=float, default=1e-5, help = "learning rate")
    parser.add_argument("--epochs", type=int, default=10, help = "training epochs")
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-4, help="privacy budget")
    parser.add_argument("--epsilon", type=float, default=10, help="privacy budget")
    parser.add_argument("--dp_type", type=str, default="text2text", choices=["text2text", "paraphrase"])
    parser.add_argument("--attack_type", type=str, default="attribute", choices=["reconstruct", "attribute"])
    parser.add_argument("--data_name", type=str, default="tweet", choices=["ClariQ", "tweet"])
    parser.add_argument("--extract_attr", type=str2bool, default=False)
    parser.add_argument("--gen_replace", type=str2bool, default=False)
    parser.add_argument("--train_generator", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    if args.data_name == "ClariQ":
        syn_dict = eval(open(f"{_ROOT_PATH}/data/attr_infer/ClariQ/prof_synonyms.txt", "r").read())
        syn_list = list(syn_dict.keys()) + sum(syn_dict.values(), [])
        male_list = ["man", "men", "male", "boy", "husband", "father", "brother"]
        female_list = ["woman", "women", "female", "girl", "lady", "wife", "mother", "sister"]
        married_list = ["married", "engaged", "dating", "boyfriend", "spouse", "girlfriend", "fiancee", "lover", "partner", "wife", "husband"]
        unmarried_list = ["single", "divorsed", "widow", "spouseless", "celibate", "unwed", "fancy-free"]
        
        data_path = f"{_ROOT_PATH}/data/attr_infer/ClariQ/train_synthetic.pkl"
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # inputs = []
        # data_paths = [f"{_ROOT_PATH}/data/attr_infer/ClariQ/train.tsv", 
        #                 f"{_ROOT_PATH}/data/attr_infer/ClariQ/dev.tsv"]
        # for data_path in data_paths:
        #     data=pd.read_csv(data_path, sep='\t')
        #     inputs += data["question"].to_list()
        #     inputs += data["answer"].to_list()
        inputs = []
        for key in data:
            inputs.append(data[key]['question'])
            inputs.append(data[key]['answer'])
        # generate gender classification data
        gender_category = []
        pre_lines = []
        for line in inputs:
            if not isinstance(line, str):
                continue
            if line in pre_lines:
                continue
            words = line.lower().split()
            male_found = False
            for kw in male_list:
                if kw in words:
                    male_found = True
                    break
            female_found = False
            for kw in female_list:
                if kw in words:
                    female_found = True
                    break
            if int(male_found) + int(female_found) == 1:
                item = {"sentence": line, "gender": "male" if male_found else "female"}
                pre_lines.append(line)
                gender_category.append(item)
        out_path = f"{_ROOT_PATH}/data/attr_infer/ClariQ/gender_classification.json"
        with open(out_path, "w") as f:
            json.dump(gender_category, f, indent=4)
            print(len(gender_category))
        
        # generate family dataset
        family_category = []
        pre_lines = []
        for line in inputs:
            if not isinstance(line, str):
                continue
            if line in pre_lines:
                continue
            words = line.lower().split()
            married_found = False
            for kw in married_list:
                if kw in words:
                    married_found = True
                    break
            unmarried_found = False
            for kw in unmarried_list:
                if kw in words:
                    unmarried_found = True
                    break
            if int(married_found) + int(unmarried_found) == 1:
                item = {"sentence": line, "family": "married" if married_found else "unmarried"}
                pre_lines.append(line)
                family_category.append(item)
        out_path = f"{_ROOT_PATH}/data/attr_infer/ClariQ/family_classification.json"
        with open(out_path, "w") as f:
            json.dump(family_category, f, indent=4)
            print(len(family_category))
    
    elif args.data_name == "tweet":
        client = OpenAI(api_key=_API_KEY)
        data_path = f"{_ROOT_PATH}/data/attr_infer/gender_dataset.csv"
        dataset = pd.read_csv(data_path, encoding='iso-8859-1')
        dataset = dataset[(dataset["gender"] != "unknown") & (~dataset["gender"].isna())]
        dataset = dataset[~dataset["text"].isna()]
        text_list = dataset["text"].to_list()
        gender_list = dataset["gender"].to_list()
        dataset = {key: value for key, value in zip(text_list, gender_list)}

        if args.extract_attr:
            cnt = 0 
            # identify the key attributes
            if "gpt-4" in args.eval_model:
                ft_dataset = []
                out_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_gptft_component.jsonl"
                max_step =  50
            else:
                args.eval_model = "ft:gpt-3.5-turbo-1106:personal:attr-extract:9NYTtOBa"
                out_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_key_component.json"
                with open(out_path) as f:
                    outputs = json.load(f)
                pre_texts = [item["text"] for item in outputs]
                max_step =  17000
            for text, gender in tqdm(dataset.items()):
                if text in pre_texts:
                    continue
                prompt = identify_attr_template.format(sentence=text)
                success = False
                n_tries = 0
                while not success:
                    components = get_response(client, prompt, args)
                    try:
                        components = eval(components)
                        success=True
                    except Exception as e:
                        n_tries += 1
                    if n_tries >= 5:
                        break
                if success:
                    if "gpt-4" in args.eval_model:
                        message = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(components)}]}
                        ft_dataset.append(message)
                        write_list(out_path, ft_dataset)
                    else:
                        item = {"text": text, "attributes": components, "label": gender}
                        outputs.append(item)
                        with open(out_path, "w") as f:
                            json.dump(outputs, f, indent=4)
                    cnt += 1
                if cnt >= max_step:
                    break
            
        # replace the key attributes
        if args.gen_replace:
            cnt = 0 
            data_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_key_component.json"
            with open(data_path) as f:
                dataset = json.load(f)

            if "gpt-4" in args.eval_model:
                ft_dataset = []
                out_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_gptft_replace.jsonl"
                max_step =  50
            else:
                args.eval_model = "ft:gpt-3.5-turbo-1106:personal:tweet-replace:9Nd9atV7"
                out_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_replacement.json"
                with open(out_path) as f:
                    outputs = json.load(f)
                pre_texts = [item["text"] for item in outputs]
                max_step =  2000
            n_replaces = 10
            for sample in tqdm(dataset):
                text, attributes, label = sample["text"], sample["attributes"], sample["label"]
                if text in pre_texts:
                    continue
                prompt = replace_review_template.format(attributes=attributes, n_replaces=n_replaces, sentence=text)
                success = False
                n_tries = 0
                while not success:
                    replacements = get_response(client, prompt, args)
                    # print(replacements)
                    try:
                        replacements = eval(replacements)
                        if len(replacements) == n_replaces:
                            success=True
                        else:
                            n_tries += 1
                    except Exception as e:
                        n_tries += 1
                    if n_tries >= 5:
                        break
                if success:
                    if "gpt-4" in args.eval_model:
                        message = {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": str(replacements)}]}
                        ft_dataset.append(message)
                        write_list(out_path, ft_dataset)
                    else:
                        item = {"text": text, "attributes": attributes, "replacement": replacements, "label": label}
                        outputs.append(item)
                        with open(out_path, "w") as f:
                            json.dump(outputs, f, indent=4)
                    cnt += 1
                if cnt >= max_step:
                    break
        
        if args.train_generator:
            data_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_replacement.json"
            with open(data_path) as f:
                dataset = json.load(f)
            inputs = []
            for sample in dataset:
                text, attributes, replacements = sample["text"], sample["attributes"], sample["replacement"]
                for rpl in replacements:
                    item = {"text": text, "attributes": attributes, "replacement": rpl}
                    inputs.append(item)
            n_replaces=20
            if args.debug:
                inputs = inputs[:100]
                args.epochs = 2
                n_replaces = 5

            tokenizer, model = get_model_tokenizer_qa(args.generator)
            train_dataset = TweetRplDataset(inputs, tokenizer)
            train_loader = DataLoader(train_dataset, shuffle=True, 
                                    collate_fn=default_data_collator, batch_size=args.train_batch_size)
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
            data_path = f"{_ROOT_PATH}/results/tweet/attack/tweet_key_component.json"
            with open(data_path) as f:
                test_dataset = json.load(f)
            
            if args.debug:
                test_dataset = test_dataset[-5:]

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

                # save final epoch of model
                if epoch == args.epochs - 1:
                    model_name = args.generator.split("/")[-1]
                    model_dir = f"{_ROOT_PATH}/save_models/replace/tweet/{model_name}"
                    if not os.path.exists(model_dir):
                        os.makedirs(model_dir, exist_ok=True)
                    model.save_pretrained(model_dir)

                outputs = []
                text2idx = {}
                idx = 0
                for sample in test_dataset:
                    text = sample["text"]
                    sample["replacement"] = []
                    outputs.append(sample)
                    text2idx[text] = idx
                    idx += 1
                # generate sample
                model_name = args.generator.split("/")[-1]
                output_dir = f"{_ROOT_PATH}/results/tweet/attack/{model_name}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                out_path = f"{output_dir}/tweet_all_replacements_epoch_{epoch}.json"
                
                for i in tqdm(range(0, len(test_dataset), args.test_batch_size)):
                    samples = test_dataset[i:(i+args.test_batch_size)]
                    prompts = []
                    for sample in samples:
                        text, attr_list, label = sample["text"], sample["attributes"], sample["label"]
                        prompt = text
                        prompt = prompt + f" {tokenizer.bos_token} " + f" {tokenizer.bos_token} ".join(attr_list)
                        prompts.append(prompt)
                    input_toks = tokenizer(prompts, max_length=512, padding=True, return_tensors="pt")
                    for _ in range(n_replaces):
                        for key in input_toks:
                            input_toks[key] = input_toks[key].to(model.device)
                        try:
                            output_ids = model.generate(**input_toks, do_sample=True, 
                                        temperature=1, max_new_tokens=args.max_new_tokens)
                        except Exception as e:
                            continue
                        sents = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                        for sample, sent in zip(samples, sents):
                            text = sample["text"]
                            idx = text2idx[text]
                            outputs[idx]["replacement"].append(sent)
                    if epoch < args.epochs - 1 and i >= 10 * args.test_batch_size:
                        break
                    if i  % (args.test_batch_size * 5) == 0:
                        with open(out_path, "w") as f:
                            json.dump(outputs, f, indent=4)
from utils.globals import *
from utils.utils import *
from utils.param import str2bool
from utils.score_utils import *

from dataset.data import AttrExtract
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
import torch.optim as optim

import pandas as pd
from tqdm import tqdm
import json
import os
import random
import argparse
import sys

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=500,
        help = "max new token for text generation")
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--eval_model", type=str, default="gpt-4o")
    parser.add_argument("--val", type=str2bool, default=False)
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gen_ft_sample", type=str2bool, default=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    _FT_SIZE = 80
    if args.gen_ft_sample:
        ft_dataset = dataset[:_FT_SIZE]
        outputs = []
        for item in ft_dataset:
            query = item["query"]
            user_prompt = extract_attr_zeroshot_template.format(query=query)
            attr = item["attribute"]

            outputs.append({"messages": [{"role": "user", "content": user_prompt}, 
                {"role": "assistant", "content": str(attr)}]})
        output_path = f"{_ROOT_PATH}/results/{args.data_name}/question_attrs_gptft.jsonl"
        write_list(output_path, outputs)
        sys.exit()

    if "gpt-4o" in args.eval_model:
        eval_mod = get_eval_model(args, args.eval_model)
        
        data_path = f"{_ROOT_PATH}/data/p2f/agg_fQnA.csv"
        dataset = pd.read_csv(data_path)
        query_list = dataset["fQ"].tolist()

        output_path = f"{_ROOT_PATH}/data/p2f/question_attrs_{args.eval_model}.json"

        question2attr = {}
        true_attrs = []
        pred_attrs = []
        # if "gpt-3.5" in args.eval_model:
        #     dataset = dataset[_FT_SIZE:]
        with tqdm(total=len(dataset), unit='batch') as pbar:
            for query in query_list:
                prompt = extract_attr_p2f_template.format(query=query)
                # print(prompt)
                success = False
                n_tries = 0
                while not success:
                    response = eval_mod.generate([prompt])[0]
                    try:
                        response = eval(response)
                        success = True
                        for attr in response:
                            if not isinstance(attr, str):
                                success = False
                        if success == False:
                            n_tries += 1
                    except Exception as e:
                        n_tries += 1
                    
                    if success or n_tries >= 5:
                        break
                
                pbar.update(1)

                if success:
                    question2attr[query] = response
                    with open(output_path, "w") as f:
                        json.dump(question2attr, f, indent=4)
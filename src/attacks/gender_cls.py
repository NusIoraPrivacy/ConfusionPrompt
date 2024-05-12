from utils.globals import *
from utils.param import str2bool

import argparse
import json
import os
import pandas as pd

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--para_model", type=str, default="eugenesiow/bart-paraphrase",
                        choices=["google/flan-t5-xl", "eugenesiow/bart-paraphrase"])
    parser.add_argument("--t2t_model", type=str, default="facebook/bart-large")
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
        help = "max new token for text generation")
    parser.add_argument("--data_name", type=str, default="tweet", choices=["ClariQ", "tweet"])
    parser.add_argument("--lr", type=float, default=1e-5, help = "learning rate")
    parser.add_argument("--epochs", type=int, default=20, help = "training epochs")
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--delta", type=float, default=1e-4, help="privacy budget")
    parser.add_argument("--epsilon", type=float, default=10, help="privacy budget")
    parser.add_argument("--dp_type", type=str, default="text2text", choices=["text2text", "paraphrase"])
    parser.add_argument("--attack_type", type=str, default="attribute", choices=["reconstruct", "attribute"])
    parser.add_argument("--gen_sample", type=str2bool, default=True)
    parser.add_argument("--gen_replacement", type=str2bool, default=False)
    parser.add_argument("--attack_flag", type=str2bool, default=True)
    parser.add_argument("--debug", type=str2bool, default=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    data_path = f"{_ROOT_PATH}/data/attr_infer/gender_dataset.csv"
    dataset = pd.read_csv(data_path, encoding='iso-8859-1')
    dataset = dataset[dataset["gender"] != "unknown"]
    text_inputs = dataset["text"].to_list()
    gender = dataset["gender"].to_list()
    print(dataset["gender"].value_counts())
    print(text_inputs[:5])
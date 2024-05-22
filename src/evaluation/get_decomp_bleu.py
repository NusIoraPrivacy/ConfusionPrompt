from utils.param import str2bool
from utils.globals import *
from utils.utils import *
from utils.score_utils import *

from dataset.get_data import *
import glob
import os
import argparse

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default=_ROOT_PATH)
    parser.add_argument("--decomp_data", type=str, default="musique")
    parser.add_argument("--base_model", type=str, default="bart_pretrained")
    parser.add_argument("--test_mode", type=str, default="sp")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    decomp_references = load_decomp_dataset(args, split="test")
    query2decomp = {}
    for item in decomp_references:
        query = item["question"]
        decomposition = item["decomposition"]
        decomposition = " <s> ".join(decomposition)
        query = format_question(query)
        query2decomp[query] = decomposition
    
    model_name = args.base_model.split("/")[-1]
    input_dir = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}"
    files = glob.glob(f"{input_dir}/decompose_{args.test_mode}_test_epoch_[0-9].json")
    for file_path in files:
        predict_inputs = read_data(file_path)
        references = []
        predictions = []
        for item in predict_inputs:
            query = item["question"]
            query = format_question(query)
            decomposition = item["decomposition"]
            if isinstance(decomposition, str):
                this_ref = query2decomp[query]
                references.append(this_ref)
                predictions.append(decomposition)
            elif isinstance(decomposition, list):
                for decomp in decomposition:
                    this_ref = query2decomp[query]
                    references.append(this_ref)
                    predictions.append(decomp)
        bleu_score = blue(predictions, references)
        data_file = file_path.split("/")[-1]
        print(f"BLEU score for {data_file}: {bleu_score}")
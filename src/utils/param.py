import argparse
from pathlib import Path
import sys
import json
import os

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def parse_args():
    json_text = Path(sys.argv[1]).read_text()
    args = argparse.Namespace(**json.loads(json_text))
    parser = argparse.ArgumentParser()
    parser.add_argument("config_json", type=str, default="configs/vit_cifar10.json")

    parser.add_argument("--root_path", type=str, default=_ROOT_PATH,
        help="root path")
    
    parser.add_argument( "--decomp_data", type=str, default="strategyQA",
        choices=["strategyQA", "musique", "hotpotqa", "hotpotqa-yn", "mix", "mmlu"],
        help="dataset to train decompose model")
    
    parser.add_argument( "--recomp_data", type=str, default="strategyQA",
        choices=["strategyQA", "musique", "hotpotqa", "mix", "mmlu"],
        help="dataset to train recompose model")
    
    parser.add_argument( "--level", type=str, default="hard",
        choices=["easy", "hard", "medium"],
        help="hardness level in hotpotqa dataset")
    
    parser.add_argument("--token_len", type=int, default=512,
        help="maximum length of input sequence")
    
    parser.add_argument("--base_model", type=str, default='bart_pretrained',
        choices = ["facebook/bart-large", 'bart_pretrained', "meta-llama/Llama-2-7b-chat-hf",
                "bert-large-uncased", "bert-base-uncased", "distilbert-base-uncased", "FacebookAI/roberta-base", 
                "FacebookAI/roberta-large", "eugenesiow/bart-paraphrase",
                "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality",
                "deepset/roberta-base-squad2", "deepset/roberta-large-squad2"],
        help="maximum length of input sequence")
    
    parser.add_argument("--test_epoch", type=int, default=3,
        help = "epoch of saved model to test")
    
    parser.add_argument("--train_batch_size", type=int, default=8,
        help = "batch size to train a model")
    
    parser.add_argument("--test_batch_size", type=int, default=5,
        help = "batch size to test a model")

    parser.add_argument("--val", type=str2bool, default=False,
        help = "whether to conduct validation during training")
    
    parser.add_argument("--redone_decomp", type=str2bool, default=False,
        help = "whether to re-extract the decomposition for the unqualified ones")
    
    parser.add_argument("--eval_complexity", type=str2bool, default=False,
        help = "whether to evaluation the complexity of decompose model")
    
    parser.add_argument("--use_attr", type=str2bool, default=False,
        help = "whether to use attributes when creating decompositions")
    
    parser.add_argument("--extract_attr", type=str2bool, default=False,
        help = "whether to extract attributes using llm")
    
    parser.add_argument("--lr", type=float, default=5e-5,
        help = "learning rate")
    
    parser.add_argument("--max_new_token", type=int, default=500,
        help = "max new token for text generation")
    
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    
    parser.add_argument("--n_resp", type=int, default=5,
        help = "responses to generate for each query")
    
    parser.add_argument("--use_peft", type=str2bool, default=False,
        help = "whether to use peft for finetining")
    
    parser.add_argument("--pretrain_recomp", type=str2bool, default=False,
        help = "whether to pretrain recomposition model")

    parser.add_argument("--lora_r", type=int,  default=8)

    parser.add_argument("--lora_alpha", type=float, default=32)

    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--ent_algo", type=str, default="textblob",
        choices = ["spacy", "flair", "bert", "textblob"],
        help = "entity recognition algorithm")

    parser.add_argument("--gpt_model", type=str, default="gpt-4",
        choices = ["gpt-4", "gpt-3.5-turbo"])

    parser.add_argument("--gpt_temperature", type=float, default=0)

    parser.add_argument("--gpt_max_tokens", type=int, default=200)

    parser.add_argument("--simple_thd", type=int, default=1,
        help = "The maximum hardness score of the last question for it to be considered simple")

    parser.add_argument("--n_replaces", type=int, default=3,
        help = "# of replacements generated for each sentence")

    parser.add_argument("--test_mode", type=str, default="dpo", 
        choices = ["dpo", "supervise"],
        help = "test the dpo or supervise model")

    parser.add_argument("--eval_model", type=str, default="gpt-4",
        choices = ["jondurbin/spicyboros-13b-2.2", "jondurbin/spicyboros-70b-2.2", "gpt-4-1106-preview",
                "gpt-4", "gpt-3.5-turbo", "lmsys/vicuna-13b-v1.5", "01-ai/Yi-34B-Chat", 
                "meta-llama/Meta-Llama-3-8B", "meta-llama/Meta-Llama-3-70B-Instruct"],
        help="model to evaluate the replaced output")

    parser.add_argument("--eval_do_sample", type=str2bool, default=True,
        help="do_sample in generate configuration")
    
    parser.add_argument("--query_gpt", type=str2bool, default=True,
        help="whether to query the gpt to obtain subanswers")
    
    parser.add_argument("--eval_qualify", type=str2bool, default=True,
        help="whether to query gpt to evaluate if each answer is qualified")

    parser.add_argument("--extract_decomp", type=str2bool, default=True,
        help="whether to extract the decompostion")

    parser.add_argument("--eval_temperature", type=float, default=1,
        help="temperature in generate configuration")

    parser.add_argument("--eval_max_new_tokens", type=int, default=50,
        help="max_new_tokens in generate configuration")

    parser.add_argument("--flu_weight", type=float, default=2,
        help="weight for the fluency score in ranking")

    parser.add_argument("--num_labels", type=int, default=4,
        help="number of labels for classification")

    parser.add_argument("--eval_batch_size", type=int, default=5,
        help="batch size to evaluate model")

    parser.add_argument("--test_data", type=str, default="strategyQA",
        choices=["strategyQA", "musique", "hotpotqa", "hotpotqa-yn"],
        help="dataset to test model")

    parser.add_argument("--mu", type=float, default=0.1,
        help="privacy budget")

    parser.add_argument("--decomp_mode", type=str, default="sp",
        choices = ["dpo", "sp"],
        help = "test the dpo or supervise decomposer")

    parser.add_argument("--gen_mode", type=str, default="sp",
        choices = ["dpo", "sp"],
        help = "test the dpo or supervise generator")

    parser.add_argument("--sim_thd", type=float, default=0.7,
        help = "threshold for similarity")
    
    parser.add_argument("--flu_thd", type=int, default=3,
        help = "threshold for fluency")
    
    parser.add_argument("--device_map", type=str, default="auto")
    
    parser.add_argument("--max_step", type=int, default=50,
        help = "maximum steps for generator evaluation")
    
    parser.add_argument("--step_by_step", type=str2bool, default=True,
        help="whether to query the model step by step")
    
    parser.add_argument("--abbrevation", type=str2bool, default=True,
        help="whether to use abbrevated sub-questions")
    
    parser.add_argument("--eval_utility", type=str2bool, default=True,
        help="whether to evaluate the utility")
    
    parser.add_argument("--use_context", type=str2bool, default=False,
        help="whether to use context in the qa dataset")
    
    parser.add_argument("--skip_eval_quality", type=str2bool, default=True,
        help="whether to skip evaluating the quality of decompositon")
    
    parser.add_argument("--gen_answer", type=str2bool, default=True)

    parser.add_argument("--debug", type=str2bool, default=False)

    args = parser.parse_args(namespace=args)

    return args

if __name__ == "__main__":
    args = parse_args()
    print(args)
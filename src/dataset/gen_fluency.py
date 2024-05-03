from utils.param import parse_args, parse_args_replace
from utils.globals import *
from utils.utils import write_list, get_eval_model

from models.causal_llm import *
from models.gpt import *

import json

from tqdm import tqdm


def create_prompt_fluency(sentences, args):
    chat_template = chat_templates[args.eval_model]
    sent_dict = {}
    for i, sent in enumerate(sentences, start=1):
        sent_dict[f"S{i}"] = sent
    sent_str = json.dumps(sent_dict)
    prompt = fluency_template.format(sentences=sent_str)
    prompt = chat_template.format(prompt=prompt)
    return prompt

if __name__ == '__main__':
    args = parse_args()
    args_replace = parse_args_replace()
    batch_size = args_replace.gen_batch_size 
    model = get_eval_model(args_replace)
    
    data_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples.json"
    dataset = []
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            dataset.append(this_item["raw query"])
            dataset.append(this_item["replacement"])
    dataset = list(set(dataset))
    
    output = []
    output_path = f"{args.root_path}/results/{args.decomp_data}/replace/fluency_trainset.json"
    with tqdm(total=int(len(dataset))/batch_size) as pbar:
        for i in range(0, len(dataset), batch_size):
            if i > 1005:
                sentences = dataset[i:(i+batch_size)]
                prompt = create_prompt_fluency(sentences, args_replace)
                prompts = [prompt]
                scores = model.generate(prompts)
                output.append({"sentences": sentences, "scores": scores[0]})
                write_list(output_path, output)
                pbar.update(1)

    # re-format output file to make each line representing a single sentence
    data_path = f"{args.root_path}/results/{args.decomp_data}/replace/fluency_trainset.json"
    dataset = []
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            dataset.append(this_item)
    
    output_path = f"{args.root_path}/results/{args.decomp_data}/replace/fluency_trainset.json"
    output = []
    for item in dataset:
        sentences, scores = item["sentences"], item["scores"]
        scores = eval(scores)
        for sent, score in zip(sentences, scores.values()):
            output.append({"sentence": sent, "score": score})
    write_list(output_path, output)
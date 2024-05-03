from utils.param import parse_args, parse_args_replace, parse_args_fluent
from utils.globals import *
from utils.utils import *

from models.causal_llm import *
from models.gpt import *

from tqdm import tqdm

import json
import os
from scipy.stats import rankdata

from sentence_transformers import SentenceTransformer, util
import torch

def create_prompt_extract_phrase(items, args):
    chat_template = chat_templates[args.eval_model]
    prompts = []
    item2prompt = {}
    item_idx = 0
    prompt_idx = 0
    for item in items:
        attributes, raw_query, replace_query = item["attributes"], item["raw query"], item["replacement"]
        item2prompt[item_idx] = []
        for attribute in attributes:
            prompt = extract_phrase_template.format(raw_query=raw_query, 
                                                    attribute=attribute, 
                                                    replace_query=replace_query)
            prompt = chat_template.format(prompt=prompt)
            prompts.append(prompt)
            item2prompt[item_idx].append(prompt_idx)
            prompt_idx +=1
        item_idx += 1
        
    return prompts, item2prompt

if __name__ == '__main__':
    args = parse_args()
    args_replace = parse_args_replace()
    args_fluent = parse_args_fluent()
    batch_size = args_replace.gen_batch_size 
    model = get_eval_model(args_replace)

    # data_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples.json"
    # dataset = read_data(data_path)
    
    # # check whether the sentence is appropriately replaced
    # output = []
    # output_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_v2.json"
    # with tqdm(total=int(len(dataset))/batch_size) as pbar:
    #     for i in range(0, len(dataset), batch_size):
    #         items = dataset[i:(i+batch_size)]
    #         prompts = create_prompt_qualify_rpl(items, args_replace)
    #         qualify_bools = model.generate(prompts)
    #         for idx, item in enumerate(items):
    #             item["qualify"] = qualify_bools[idx]
    #             output.append(item)
    #         write_list(output_path, output) 
    #         pbar.update(1)
    
    # # extract the correspondence attributes
    # data_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_v2.json"
    # dataset = []
    # with open(data_path) as dataset_file:
    #     for line in dataset_file:
    #         this_item = json.loads(line)
    #         if "yes" in this_item["qualify"].lower():
    #             dataset.append(this_item)

    # output = []
    # output_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_v3.json"
    # with tqdm(
    #         total=int(len(dataset))/batch_size
    #     ) as pbar:
    #     for i in range(0, len(dataset), batch_size):
    #         items = dataset[i:(i+batch_size)]
    #         prompts, item2prompt = create_prompt_extract_phrase(items, args_replace)
    #         phrases = model.generate(prompts)
    #         for idx, item in enumerate(items):
    #             this_phrases = []
    #             phrase_idxs = item2prompt[idx]
    #             for phrase_idx in phrase_idxs:
    #                 this_phrases.append(phrases[phrase_idx])
    #             item["replace phrases"] = this_phrases
    #             output.append(item)
        
    #         write_list(output_path, output)
    #         pbar.update(1)
    
    # # compute the cosine similarity between replaced attributes and original attributes
    # data_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_v3.json"
    # dataset = read_data(data_path)
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # output_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_v4.json"
    # output = []
    # with tqdm(total=len(dataset)) as pbar:
    #     for item in dataset:
    #         attributes, rpl_attrs = item["attributes"], item["replace phrases"]
    #         cos_sims = []
    #         for attr, rpl_attr in zip(attributes, rpl_attrs):
    #             emb1 = model.encode(attr)
    #             emb2 = model.encode(rpl_attr)
    #             cos_sim = util.cos_sim(emb1, emb2)
    #             cos_sims.append(cos_sim[0][0].item())
    #         item["similarity"] = cos_sims
    #         output.append(item)
    #         write_list(output_path, output)
    #         pbar.update(1)
    
    # # compute the fluency rating for each replaced sentence
    # data_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_v4.json"
    # dataset = read_data(data_path)
    # model_name = args_fluent.base_model.split("/")[-1]
    # model_dir = f"{args.root_path}/save_models/discriminator/{model_name}"
    # tokenizer, model = get_model_tokenizer_cls(model_dir, args_fluent.num_labels)
    # output_path = data_path
    # output = []
    # with tqdm(total=len(dataset)) as pbar:
    #     for item in dataset:
    #         rpl_sent = item["replacement"]
    #         inputs = tokenizer(rpl_sent, return_tensors="pt")
    #         for key in inputs:
    #             inputs[key] = inputs[key].to(model.device)
    #         with torch.no_grad():
    #             outputs = model(**inputs)
    #         logits = outputs.logits
    #         y_pred = torch.argmax(logits, -1)
    #         item["fluency"] = y_pred[0].item()
    #         output.append(item)
    #         pbar.update(1)
    #     write_list(output_path, output)
    
    # rank the comparisons based on the similarity and fluency score
    data_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_v4.json"
    dataset = read_data(data_path)
    output_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_final.json"
    pre_query = ""
    pre_attrs = []
    cur_item = None
    output = []
    for item in dataset:
        query, attrs, cos_sims, fluency = item["raw query"], item["attributes"], item["similarity"], item["fluency"]
        cos_sim = max(cos_sims)
        # normalize the fluency score between 0 to 1
        fluency_norm = fluency/(args_fluent.num_labels-1)
        score =  cos_sim - args_replace.flu_weight * fluency_norm
        if query == pre_query and set(attrs) == set(pre_attrs):
            cur_item["replacement"].append(item["replacement"])
            cur_item["similarity"].append(cos_sim)
            cur_item["fluency"].append(fluency)
            cur_item["score"].append(score)
        else:
            if cur_item is not None:
                pre_item = cur_item
                # rank the outputs based on the score
                score_rank = list(rankdata(pre_item["score"], method='min'))
                score_rank = [int(r) for r in score_rank]
                pre_item["rank"] = score_rank
                output.append(pre_item)
            cur_item = {"raw query": query, "attributes": attrs}
            cur_item["replacement"] = [item["replacement"]]
            cur_item["similarity"] = [cos_sim]
            cur_item["fluency"] = [fluency]
            cur_item["score"] = [score]
        pre_query = query
        pre_attrs = attrs
    # rank the outputs based on the score
    score_rank = list(rankdata(cur_item["score"], method='min'))
    score_rank = [int(r) for r in score_rank]
    cur_item["rank"] = score_rank
    output.append(cur_item)
    write_list(output_path, output)
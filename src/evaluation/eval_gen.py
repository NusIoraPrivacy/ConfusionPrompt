from utils.param import parse_args
from utils.utils import *
from utils.globals import *

from dataset.data import EvalReplace
from tqdm import tqdm

from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer, util
import numpy as np
import time

def format_replace(dataset):
    outputs = []
    for i, item in enumerate(dataset):
        question, decompositions, attrs2decomp = item["question"], item["decomposition"], item["has_attrs"]
        decomp_list = decompositions[:-1]
        # create a dictionary for decomposition to attribute
        decomp2attrs = {idx:[] for idx in range(len(decomp_list))}
        for attr in attrs2decomp:
            decomp_string = attrs2decomp[attr]
            decomp_idx = process_has_attr(decomp_string)
            for idx in decomp_idx:
                if idx <= len(decomp_list):
                    decomp2attrs[idx-1].append(attr)
        # extract the attributes and the relevant decompositions
        for idx in decomp2attrs:
            query = decomp_list[idx]
            attrs = decomp2attrs[idx]
            if len(attrs) > 0:
                item = {"raw query": query, "attributes": attrs, "question": question, "index": i}
                outputs.append(item)
    return outputs

def seq_collate_fn(seq_list):
    seq_keys = ["index", "question"]
    tensor_keys = ["input_ids", "attention_mask"]
    batch = {key:[] for key in seq_keys+tensor_keys}
    for seq_dict in seq_list:
        for key in seq_keys:
            batch[key].append(seq_dict[key])
    for key in tensor_keys:
        batch[key] = torch.stack([f[key] for f in seq_list])
    return batch

def create_prompt_extract_phrase(item, args):
    chat_template = chat_templates[args.eval_model]
    prompts = []
    attributes, raw_query, replace_query = item["attributes"], item["raw query"], item["replacement"]
    for attribute in attributes:
        prompt = extract_phrase_template.format(raw_query=raw_query, 
                                                attribute=attribute, 
                                                replace_query=replace_query)
        prompt = chat_template.format(prompt=prompt)
        prompts.append(prompt)
        
    return prompts

if __name__ == '__main__':
    args = parse_args()
    # load dataset
    model_name = args.decomp_model.split("/")[-1]
    data_path = f"{args.root_path}/results/{args.data_name}/decomp/{model_name}/decompose_{args.decomp_mode}_eval_v3.json"
    dataset = read_data(data_path)
    dataset = format_replace(dataset)

    # load model
    model_name = args.replace_model.split("/")[-1]
    model_dir = f"{args.root_path}/save_models/replace/{args.data_name}/{arg.gen_mode}/{model_name}/final"
    tokenizer, model = get_model_tokenizer(model_dir, args)
    dataset = EvalReplace(dataset, tokenizer)
    dataloader = DataLoader(
            dataset, 
            batch_size=1, 
            collate_fn=seq_collate_fn, 
            pin_memory=True,
            shuffle=True
        )
    eva_model = get_eval_model(args)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    model_name = args.disriminator.split("/")[-1]
    model_dir = f"{args.root_path}/save_models/discriminator/{model_name}"
    flu_tokenizer, flu_model = get_model_tokenizer_cls(model_dir, args.num_labels)

    generation_config = GenerationConfig(
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # inference with model and dataset
    n_resp = int(1/args.mu)
    n_resp = 1
    counts = []
    rpl_samples = []
    output_path = f"{args.root_path}/results/{args.data_name}/replace_{args.decomp_mode}_{args.gen_mode}_samples.json"
    with tqdm(total=len(dataloader), unit='batch') as pbar:
        times = []
        for step, batch in enumerate(dataloader):
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(model.device)
            prompts = tokenizer.batch_decode(batch["input_ids"])
            qualfy_cnt = 0
            total_cnt = 0
            while qualfy_cnt < n_resp:
                t1 = time.time()
                total_cnt += 1
                output_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        generation_config = generation_config,
                    )
                responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                attr_query = extract_attr_query(prompts, tokenizer)[0]
                questions = batch["question"]
                indexes= batch["index"]
                item = {"attributes": attr_query["attributes"],
                        "raw query": attr_query["raw query"],
                        "replacement": responses[0]}
                # compute the fluency rating for each replaced sentence
                rpl_sent = item["replacement"]
                inputs = flu_tokenizer(rpl_sent, return_tensors="pt")
                for key in inputs:
                    inputs[key] = inputs[key].to(flu_model.device)
                with torch.no_grad():
                    outputs = flu_model(**inputs)
                logits = outputs.logits
                y_pred = torch.argmax(logits, -1)
                flu_score = y_pred[0].item()
                if flu_score < args.flu_thd:
                    continue
                # evaluate the quality of generation
                qualify_prompts = create_prompt_qualify_rpl([item], args)
                qualify_bools = eva_model.generate(qualify_prompts)
                print(qualify_bools)
                if "yes" not in qualify_bools[0].lower():
                    continue
                # evaluate the similarity to original attributes
                sim_prompts = create_prompt_extract_phrase(item, args)
                phrases = eva_model.generate(sim_prompts)
                print(phrases)
                max_sim = 0
                for attr, rpl_attr in zip(attr_query["attributes"], phrases):
                    emb1 = sim_model.encode(attr)
                    emb2 = sim_model.encode(rpl_attr)
                    cos_sim = util.cos_sim(emb1, emb2)
                    cos_sim = cos_sim[0][0].item()
                    max_sim = max(max_sim, cos_sim)
                if max_sim > args.sim_thd:
                    continue
                qualfy_cnt += 1
                t2 = time.time()
                rpl_samples.append(item)
                write_list(output_path, rpl_samples)
            counts.append(total_cnt)
            times.append(t2-t1)
            avg_cnt = np.mean(counts)
            avg_time = np.mean(times)
            pbar.update(1)
            pbar.set_postfix(count=avg_cnt)
            if step > args.max_step:
                break
            
    print(f"Average query times for mu={args.mu}: {avg_cnt}")
    print(f"Average time cost per query: {avg_time}")
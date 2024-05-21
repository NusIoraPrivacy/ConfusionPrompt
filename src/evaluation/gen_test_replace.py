from utils.param import parse_args
from dataset.data import ReplaceDataset
from utils.utils import get_model_tokenizer, write_list
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
from transformers import (default_data_collator,
                          AutoModelForCausalLM, 
                          GenerationConfig,
                          AutoModelForSeq2SeqLM)
from peft import PeftModel
import os
from utils.utils import extract_attr_query

def test_replace(args, model, tokenizer, epoch, test_dataset=True):
    if test_dataset:
        data_path = f"{args.root_path}/data/{args.decomp_data}/replace_test_samples_demo.json"
        replace_inputs = []
        with open(data_path) as dataset_file:
            for line in dataset_file:
                this_item = json.loads(line)
                replace_inputs.append(this_item)
    else:
        data_path = f"{args.root_path}/data/{args.decomp_data}/replace_train_samples.json"
        replace_inputs = []
        with open(data_path) as dataset_file:
            for idx, line in enumerate(dataset_file):
                if idx % args.n_replaces == 0:
                    this_item = json.loads(line)
                    replace_inputs.append(this_item)
    
    rpl_dataset = ReplaceDataset(replace_inputs, tokenizer)
    test_dataloader = DataLoader(
            rpl_dataset, 
            batch_size=args.test_batch_size, 
            collate_fn=default_data_collator, 
            pin_memory=True,
            )
    generation_config = GenerationConfig(
            do_sample=True,
            temperature=args.temperature,
            max_new_tokens=args.max_new_token,
            pad_token_id=tokenizer.pad_token_id,
        )
    # generate decomposed questions with trained model
    output_list = []
    with tqdm(total=len(test_dataloader), unit='batch') as pbar:

        for step, batch in enumerate(test_dataloader):
            # just query gpt when epoch == 0
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            
            for i in range(args.n_resp):
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
            # store the file
            model_name = args.base_model.split("/")[-1]
            output_dir = f"{args.root_path}/results/{args.decomp_data}/replace/{model_name}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/replace_{args.test_mode}_epoch_{epoch}.json"
            write_list(output_path, output_list)
            pbar.update(1)

if __name__ == '__main__':
    args = parse_args()
    model_name = args.base_model.split("/")[-1]
    # model_dir = f"{args.root_path}/save_models/dpo/replace/{model_name}/final"
    model_dir = f"{args.root_path}/save_models/replace/strategyQA/bart-large/final"
    tokenizer, model = get_model_tokenizer(model_dir, args)
    # model_dir = f"{args.root_path}/save_models/replace/{args.decomp_data}/{model_name}/epoch_{args.test_epoch}"
    if args.peft is not None:
        model = PeftModel.from_pretrained(model, model_dir)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="auto")
    # model.eval()
    test_replace(args, model, tokenizer, args.test_epoch, test_dataset=False)
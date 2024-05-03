from accelerate import init_empty_weights
from utils.param import parse_args
from dataset.data import DecompDataset
from utils.utils import get_model_tokenizer, process_response, write_list
from dataset.get_data import *
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import (default_data_collator, AutoModelForCausalLM, 
                        GenerationConfig, AutoModelForSeq2SeqLM)
from peft import PeftModel
import os

def test_decomp(args, model, tokenizer, epoch, mode="test"):
    decomp_inputs = load_decomp_dataset(args, split=mode)
    test_dataset = DecompDataset(decomp_inputs, tokenizer, args.token_len)
    test_dataloader = DataLoader(
            test_dataset, 
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
    with tqdm(
        total=int(len(test_dataset)/args.test_batch_size), unit='batch'
    ) as pbar:

        for step, batch in enumerate(test_dataloader):
            # just query gpt when epoch == 0
            for key in batch.keys():
                batch[key] = batch[key].to(model.device)
            
            for i in range(args.n_resp):
                output_ids = model.generate(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        generation_config = generation_config,
                    )
                # responses = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                responses = tokenizer.batch_decode(output_ids)
                questions = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
                responses = process_response(responses, questions, tokenizer)
                for query, resp in zip(questions, responses):
                    output_list.append({"question":query, "decomposition": resp})
                output_list = sorted(output_list, key=lambda d: d["question"]) 
            # store the file
            model_name = args.base_model.split("/")[-1]
            output_dir = f"{args.root_path}/results/{args.decomp_data}/decomp/{model_name}"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            output_path = f"{output_dir}/decompose_{args.test_mode}_{mode}_epoch_{epoch}.json"
            write_list(output_path, output_list)
            pbar.update(1)

if __name__ == '__main__':
    args = parse_args()
    tokenizer, model = get_model_tokenizer(args.base_model, args)
    model_name = args.base_model.split("/")[-1]
    if args.test_mode == "dpo":
        model_dir = f"{args.root_path}/save_models/decomp/{args.test_mode}/{args.decomp_data}/{model_name}/final"
    else:
        if not "pretrained" in args.base_model:
            model_dir = f"{args.root_path}/save_models/decomp/{args.test_mode}/{args.decomp_data}/{model_name}/epoch_{args.test_epoch}"
    if not "pretrained" in args.base_model or args.test_mode == "dpo":
        if args.peft:
            model = PeftModel.from_pretrained(model, model_dir)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, device_map="auto")
    model.eval()
    test_decomp(args, model, tokenizer, 0)
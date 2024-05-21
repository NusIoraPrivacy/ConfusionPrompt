from utils.globals import *
from utils.utils import *
from utils.param import str2bool
from utils.score_utils import *

from dataset.data import AttrExtract
from torch.utils.data import DataLoader
from transformers import default_data_collator, get_scheduler
import torch.optim as optim

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
    parser.add_argument( "--data_name", type=str, default="musique", choices=["strategyQA", "musique"])
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=100,
        help = "max new token for text generation")
    parser.add_argument("--train_batch_size", type=int, default=10)
    parser.add_argument("--test_batch_size", type=int, default=10)
    parser.add_argument("--eval_model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--val", type=str2bool, default=False)
    parser.add_argument("--skip_train", type=str2bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gen_ft_sample", type=str2bool, default=False)
    args = parser.parse_args()

    return args


def load_query(data_name):
    if data_name == "strategyQA":
        data_path = f"{_ROOT_PATH}/results/{data_name}/replace/question_attrs.json"
        with open(data_path) as f:
            dataset = json.load(f)
        output = []
        for query, attribute in dataset.items():
            output.append({"query": query, "attribute": attribute})
    if data_name == "musique":
        data_path = f"{_ROOT_PATH}/data/{data_name}/musique_ans_v1.0_dev.jsonl"
        with open(data_path) as f:
            output = []
            for line in f:
                this_item = json.loads(line)
                query = this_item["question"]
                output.append({"query": query, "attribute": ""})
        # data_path = f"{_ROOT_PATH}/results/{data_name}/question_attrs.json"
        # with open(data_path) as f:
        #     dataset = json.load(f)
        # output = []
        # for query, attribute in dataset.items():
        #     output.append({"query": query, "attribute": attribute})
    return output

if __name__ == "__main__":
    args = parse_args()

    dataset = load_query(args.data_name)
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

    if "gpt" in args.eval_model:
        if "gpt-3.5" in args.eval_model:
            eval_model_map = {"strategyQA": "ft:gpt-3.5-turbo-1106:personal::9Qupj5TL",
                            "musique": "ft:gpt-3.5-turbo-1106:personal:musique:9QzPSFrm"}
            args.eval_model = eval_model_map[args.data_name]
            query_template = extract_attr_zeroshot_template
        else:
            query_template = extract_attr_template
        eval_mod = get_eval_model(args, args.eval_model)
        
        
        output_path = f"{_ROOT_PATH}/results/{args.data_name}/question_attrs_{args.eval_model}.json"
        if args.skip_train:
            with open(output_path) as f:
                pred_question2attr = json.load(f)

            true_attrs = []
            pred_attrs = []
            for item in dataset:
                query = item["query"]
                attr_list = item["attribute"]
                attr_list.sort()
                this_attr = " ".join(attr_list)
                true_attrs.append(this_attr)
                pred_attr_list = pred_question2attr[query]
                pred_attr_list.sort()
                pred_attr = " ".join(pred_attr_list)
                pred_attrs.append(pred_attr)
            
            rougeL = rouge(pred_attrs, true_attrs)
            f1 = f1_score(pred_attrs, true_attrs)
            em = exact_match_score(pred_attrs, true_attrs)
            print(f"rougeL: {rougeL}, F1 score: {f1}, exact match: {em}")
            
        else:
            question2attr = {}
            true_attrs = []
            pred_attrs = []
            # if "gpt-3.5" in args.eval_model:
            #     dataset = dataset[_FT_SIZE:]
            with tqdm(total=len(dataset), unit='batch') as pbar:
                for item in dataset:
                    query = item["query"]
                    prompt = query_template.format(query=query)
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

                        if args.val:
                            this_attr = item["attribute"]
                            this_attr.sort()
                            this_attr = " ".join(this_attr)
                            true_attrs.append(this_attr)
                            response.sort()
                            pred_attr = " ".join(response)
                            pred_attrs.append(pred_attr)
                            rougeL = rouge(pred_attrs, true_attrs)
                            f1 = f1_score(pred_attrs, true_attrs)
                            em = exact_match_score(pred_attrs, true_attrs)
                            pbar.set_postfix(rougeL=rougeL, f1=f1, exact_match=em)

            if args.val:
                print(f"rougeL: {rougeL}, F1 score: {f1}, exact match: {em}")
    
    else:
        tokenizer, model = get_model_tokenizer_qa(args.eval_model, device_map="auto")
        if args.data_name == "musique":
            addition_dataset = load_query("strategyQA")
            random.shuffle(addition_dataset)
        if args.val:
            n_trains = int(len(dataset) * 0.8)
            random.shuffle(dataset)
            train_inputs = dataset[:n_trains]
            if args.data_name == "musique":
                train_inputs = train_inputs + addition_dataset[:300]
            test_inputs = dataset[n_trains:]
        else:
            train_inputs = dataset
        # train_inputs = train_inputs[:5]
        train_dataset = AttrExtract(train_inputs, tokenizer)
        train_dataloader = DataLoader(
                train_dataset, 
                batch_size=args.train_batch_size, 
                collate_fn=default_data_collator, 
                pin_memory=True,
                shuffle=True
            )
        if args.val:
            test_dataset = AttrExtract(test_inputs, tokenizer)
            test_dataloader = DataLoader(
                    test_dataset, 
                    batch_size=args.test_batch_size, 
                    collate_fn=default_data_collator, 
                    pin_memory=True,
                    shuffle=True
                )
        optimizer = optim.AdamW(
                    model.parameters(),
                    lr=args.lr,
                    weight_decay=0.0,
                )
        num_training_steps = args.epochs * len(train_dataloader )
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps,
            )
        model_name = args.eval_model.split("/")[-1]
        output_path = f"{_ROOT_PATH}/results/{args.data_name}/question_attrs_{model_name}.json"
        best_scores = {"f1": 0, "rougeL": 0, "exact match": 0}
        for epoch in range(args.epochs):
            model.train()
            loss_list = []
            with tqdm(total=len(train_dataloader), unit='batch') as pbar:
                for step, batch in enumerate(train_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(model.device)
                    output = model(**batch) 
                    loss = output.loss
                    loss_list.append(loss.item())
                    loss.backward()
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())
                print(f'[epoch: {epoch}] Loss: {np.mean(np.array(loss_list))}')
            
            if args.val:
                generation_config = GenerationConfig(
                        do_sample=True,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                pred_question2attr = {}
                pred_attr_list = []
                true_attr_list = []
                with tqdm(total=len(test_dataloader), unit='batch') as pbar:
                    for step, batch in enumerate(test_dataloader):
                        for key in batch.keys():
                            batch[key] = batch[key].to(model.device)
                        
                        output_ids = model.generate(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            generation_config = generation_config,
                        )
                        questions = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

                        # obtain true attributes
                        label_attrs = []
                        target_ids = batch["labels"]
                        for target in target_ids:
                            target = target[target != IGNORE_INDEX]
                            attr_str = tokenizer.decode(target)
                            label_attrs.append(attr_str)
                        label_attrs = process_response(label_attrs, questions, tokenizer)
                        responses = tokenizer.batch_decode(output_ids)
                        
                        responses = process_response(responses, questions, tokenizer)
                        for query, pred_attr, true_attr in zip(questions, responses, label_attrs):
                            pred_attr = pred_attr.split(tokenizer.bos_token)
                            pred_attr = [i.strip() for i in pred_attr]
                            query = format_question(query)
                            pred_question2attr[query] = pred_attr
                            pred_attr.sort()
                            pred_attr_list.append(" ".join(pred_attr))
                            true_attr = true_attr.split(tokenizer.bos_token)
                            true_attr = [i.strip() for i in true_attr]
                            true_attr.sort()
                            true_attr_list.append(" ".join(true_attr))
                        with open(output_path, "w") as f:
                            json.dump(pred_question2attr, f, indent=4)

                rougeL = rouge(pred_attr_list, true_attr_list)
                f1 = f1_score(pred_attr_list, true_attr_list)
                em = exact_match_score(pred_attr_list, true_attr_list)
                print(f"Epoch: {epoch}, rougeL: {rougeL}, F1 score: {f1}, exact match: {em}")
                if f1 > best_scores["f1"]:
                    best_scores["f1"], best_scores["rougeL"], best_scores["exact match"]= f1, rougeL, em

        model_name = args.eval_model.split("/")[-1]
        model_dir = f"{_ROOT_PATH}/save_models/extract_attr/{args.data_name}/{model_name}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        print(best_scores)
    
    
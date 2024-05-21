from transformers import (AutoTokenizer, 
                          AutoModel, GPT2Tokenizer, GPT2Model, 
                          OPTForSequenceClassification,
                          AutoModelForCausalLM, 
                          AutoModelForSeq2SeqLM,
                          BertLMHeadModel,
                          DistilBertForSequenceClassification,
                          LlamaForSequenceClassification,
                          BertForSequenceClassification,
                          GPT2ForSequenceClassification,
                          BartForSequenceClassification,
                          RobertaForSequenceClassification,
                          T5ForConditionalGeneration,
                          RobertaForCausalLM)
from collections import OrderedDict
import torch
from peft import get_peft_model, LoraConfig, TaskType
from utils.globals import *
from models.causal_llm import *
from models.gpt import *
import json
import os
import time
import re
import tiktoken
from openai import (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    APIError,
)

def get_model_tokenizer(model_name, args=None, device_map="auto"):
    if model_name in ("THUDM/chatglm2-6b-int4", "THUDM/chatglm2-6b"):
        base_model = AutoModel.from_pretrained(model_name, trust_remote_code=True) # FP16 by default
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    elif 'gpt2' in model_name:
        try:
            tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        except:
            tokenizer = GPT2Tokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2Model.from_pretrained(model_name, device_map=device_map)
    elif 'opt' in model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = OPTForSequenceClassification.from_pretrained(model_name, device_map=device_map)
    elif 'llama' in model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map)
    elif "bart_pretrained" in model_name:
        tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
        base_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large", return_dict=True, device_map=device_map)
        state_dict = torch.load(f"{args.root_path}/save_models/decomp/bart_pretrained/weights.th")
        state_dict = OrderedDict([(k.replace('_seq2seq.',''), v) for k, v in state_dict.items()])
        base_model.load_state_dict(state_dict, strict=False)
    elif "bart-" in model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name,)
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model =  AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map)
    elif "bert" in model_name:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name,)
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base_model = BertLMHeadModel.from_pretrained(model_name).cuda()

    if args:
        if args.use_peft:
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                inference_mode=False, 
                r=args.lora_r, 
                lora_alpha=args.lora_alpha, 
                lora_dropout=args.lora_dropout
                )
            base_model = get_peft_model(base_model, peft_config)
    return tokenizer, base_model

def get_model_tokenizer_cls(model_name, num_labels, args=None, device_map="auto"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if 'gpt2' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        base_model = GPT2ForSequenceClassification.from_pretrained(model_name, device_map=device_map)
    elif 'opt' in model_name:
        base_model = OPTForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, device_map=device_map)
    elif 'llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        base_model = LlamaForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, device_map=device_map)
    elif "bart" in model_name:
        base_model = BartForSequenceClassification.from_pretrained(model_name, 
                                                                num_labels=num_labels, device_map=device_map)
    elif "roberta" in model_name:
        base_model = RobertaForSequenceClassification.from_pretrained(model_name, 
                                                        num_labels=num_labels)
        base_model = base_model.cuda()
    elif "distilbert" in model_name:
        base_model = DistilBertForSequenceClassification.from_pretrained(model_name, 
                                                        num_labels=num_labels)
        base_model = base_model.cuda()
    elif "bert" in model_name:
        base_model = BertForSequenceClassification.from_pretrained(model_name, 
                                                        num_labels=num_labels)
        base_model = base_model.cuda()
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token
        tokenizer.eos_token_id = tokenizer.pad_token_id
    return tokenizer, base_model

def get_model_tokenizer_qa(model_name, args=None, device_map="auto"):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if 't5' in model_name:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map)
    elif 'llama' in model_name:
        tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(model_name, num_labels=num_labels, device_map=device_map)
    elif "bart" in model_name:
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map)
    elif "roberta" in model_name:
        base_model = RobertaForCausalLM.from_pretrained(model_name, device_map=device_map, is_decoder=True)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = tokenizer.pad_token
        tokenizer.eos_token_id = tokenizer.pad_token_id
    return tokenizer, base_model

def decomp2list(decompositions, args, simple_last = '5'):
    # convert string of sub-questions into a list
    decomp_list = decompositions.split("<s>")
    temp = []
    for decomp in decomp_list:
        if len(decomp) > 1:
            temp.append(decomp)
    decomp_list = temp
    # if the last question is simple, then it's answered by local re-composition model
    if int(simple_last) <= args.simple_thd:
        decomp_list = decomp_list[:-1]
    return decomp_list

def sing2multi(idx_keys, comb_keys, inputs):
    inputs = sorted(inputs, key=lambda d: tuple([d[key] for key in idx_keys])) 
    output = []
    pre_val_dict = {key: "" for key in idx_keys}
    curr_item = None
    for item in inputs:
        equal = True
        for key in idx_keys:
            if isinstance(item[key], list):
                this_equal = (set(pre_val_dict[key]) == set(item[key]))
            else:
                this_equal =  pre_val_dict[key] == item[key]
            equal = (equal & this_equal)
        if equal:
            for key in comb_keys:
                curr_item[key].append(item[key])
        else:
            if curr_item is not None:
                output.append(curr_item)
            curr_item = {}
            for key in idx_keys:
                curr_item[key] = item[key]
                pre_val_dict[key] = item[key]
            for key in comb_keys:
                curr_item[key] = [item[key]]
    output.append(curr_item)
    return output

def read_data(data_path):
    # read data in the form of list of dictionaries
    dataset = []
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            dataset.append(this_item)
    return dataset

def process_response(responses, questions, tokenizer):
    output = []
    for resp, query in zip(responses, questions):
        resp = resp.replace(query, "")
        if tokenizer.sep_token is not None:
            resp = resp.replace(tokenizer.sep_token, "")
        resp = resp.replace(tokenizer.pad_token, "")
        resp = resp.strip(tokenizer.bos_token)
        resp = resp.strip()
        output.append(resp)
    return output

def write_list(file_path, output):
    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)
    with open(file_path, "w") as f:
        for examp in output:
            json.dump(examp, f) 
            f.write("\n")

def get_response(client, messages, args):
    """
    Obtain response from GPT
    """
    SLEEP_TIME = 10
    success = False
    cnt = 0
    while not success:
        if cnt >= 50:
            rslt = "Error"
            break
        try:
            response = client.chat.completions.create(
                model=args.gpt_model,
                messages=messages,
                temperature=args.gpt_temperature,
                max_tokens=args.gpt_max_tokens,
                # frequency_penalty=frequency_penalty,
                # presence_penalty=presence_penalty,
            )
            rslt = response.choices[0].message.content
            success = True
        except RateLimitError as e:
            print(f"sleep {SLEEP_TIME} seconds for rate limit error")
            time.sleep(SLEEP_TIME)
        except APITimeoutError as e:
            print(f"sleep {SLEEP_TIME} seconds for time out error")
            time.sleep(SLEEP_TIME)
        except APIConnectionError as e:
            print(f"sleep {SLEEP_TIME} seconds for api connection error")
            time.sleep(SLEEP_TIME)
        except APIError as e:
            print(f"sleep {SLEEP_TIME} seconds for api error")
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print(e)
            success = True
            rslt = "Error"
        cnt += 1
    return rslt

def process_candidate(candidates):
    """
    convert string of "[A, B, D]" to list of [0, 1, 3]
    """
    def char2num(candidates):
        indexes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        # print(candidates)
        candidates = candidates.strip(" ")
        candidates = candidates.strip("[|]")
        candidates = re.split(r"/|,", candidates)
        output = []
        for ltr in candidates:
            ltr = ltr.strip()
            ltr = ltr.strip("'|'|\"|\"")
            ltr = ltr.strip(".")
            if len(ltr) > 0:
                try:
                    this_idx = indexes.index(ltr)
                    output.append(this_idx)
                except Exception as e:
                    print(ltr)
        return output
    output = char2num(candidates)
    if len(output) == 0:
        results = re.search("\[.+?\]", candidates)
        if results:
            results = results.group()
            output = char2num(candidates)
            print(output)
    return output

def process_has_attr(decomp_string):
    """
    convert string of "[1, 2, 3]" to list of [1, 2, 3]
    """
    decomp_string = decomp_string.strip("[|]")
    decomp_list = decomp_string.split(",")
    output = []
    for idx in decomp_list:
        idx = idx.strip()
        if len(idx) > 0:
            try:
                output.append(int(idx))
            except Exception as e:
                pass
    return output

def create_message_qualify(question, decompositions):
    indexes = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    template = qualify_template
    cnt = 0
    choices = ""
    for decomp in decompositions:
        this_idx = indexes[cnt]
        this_choice = this_idx + "."+ decomp + '\n'
        choices += this_choice
        cnt += 1
    prompt = template.format(question = question, choices = choices)
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages

def create_prompt_qualify_rpl(items, args):
    chat_template = chat_templates[args.eval_model]
    prompts = []
    for item in items:
        attributes, raw_query, replace_query = item["attributes"], item["raw query"], item["replacement"]
        attr_str = ""
        for i, attr in enumerate(attributes):
            if i == 0:
                attr_str += f"\"{attr}\""
            elif i == len(attributes) - 1:
                attr_str += f" and \"{attr}\""
            else:
                attr_str += f", \"{attr}\""
        prompt = qualify_replace_template.format(raw_query=raw_query, 
                                        attributes=attr_str, 
                                        replace_query=replace_query)
        prompt = chat_template.format(prompt=prompt)
        prompts.append(prompt)
    return prompts

def extract_attr_query(prompts, tokenizer):
    """
    extract the attributes and raw querys from the prompt
    """
    outputs = []
    for prompt in prompts:
        this_item = {}
        prompt = prompt.replace(tokenizer.eos_token, "")
        prompt = prompt.replace(tokenizer.pad_token, "")
        prompt_list = prompt.split(f" {tokenizer.bos_token} ")
        attr_list, sentence = prompt_list[:-1], prompt_list[-1]
        sentence = sentence.replace(tokenizer.bos_token, "")
        sentence = format_question(sentence)
        this_item["raw query"] = sentence
        # attributes = attributes.replace(f"Attributes {tokenizer.bos_token} ", "")
        # attr_list = attributes.split(tokenizer.bos_token)
        this_item["attributes"] = []
        for attr in attr_list:
            attr = attr.replace(tokenizer.bos_token, "")
            attr = attr.strip()
            if len(attr) > 0:
                this_item["attributes"].append(attr)
        outputs.append(this_item)
    return outputs

def get_eval_model(args_eval, model_name = None):
    eval_model = model_name if model_name is not None else args_eval.eval_model
    if "spicyboros" in eval_model:
        model = SpicyBoros(eval_model, 
                        do_sample=args_eval.do_sample,
                        temperature=args_eval.temperature,
                        max_new_tokens=args_eval.max_new_tokens)
    elif "vicuna" in eval_model:
        model = Vicuna(eval_model, 
                        do_sample=args_eval.do_sample,
                        temperature=args_eval.temperature,
                        max_new_tokens=args_eval.max_new_tokens)
    elif "Yi" in eval_model:
        model = YiChat(eval_model, 
                        do_sample=args_eval.do_sample,
                        temperature=args_eval.temperature,
                        max_new_tokens=args_eval.max_new_tokens)
    elif "llama" in eval_model:
        model = YiChat(eval_model, 
                        do_sample=args_eval.do_sample,
                        temperature=args_eval.temperature,
                        max_new_tokens=args_eval.max_new_tokens)
    elif "gpt" in eval_model:
        model = ChatGPT(eval_model, args_eval)
    return model

def format_question(question):
    question = question.strip()
    question = question.strip("?")
    question = question.strip()
    question = question + "?"
    return question

def extract_attribute(dataset, args):
    # construct the reference dataset
    question2attr = {}
    data_path = f"{args.root_path}/results/{args.decomp_data}/replace/question_attrs.json"
    with open(data_path) as f:
        question2attr = json.load(f)
    # for item in ref_dataset:
    #     question = format_question(item["question"])
    #     attributes = item["private attributes"]
    #     question2attr[question] = attributes
    # add attributes to the dataset
    output = []
    for sample in dataset:
        question = format_question(sample["question"])
        decompostions = sample["decomposition"]
        subanswers = sample["subanswer"]
        for decomp, subans in zip(decompostions, subanswers):
            if len(subans) > 0:
                decomp_list = decomp.split("<s>")
                decomp_list = [sent.strip() for sent in decomp_list]
                attr = question2attr[question]
                item = {
                    "question": question,
                    "decomposition": decomp_list,
                    "attributes": attr,
                    "subanswer": subans
                }
                output.append(item)
    return output

def num_tokens_from_string(string):
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
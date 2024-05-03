from datasets import Dataset
import torch
import json
import copy
from utils.globals import (IGNORE_INDEX, chat_templates, dataset_type,
                    direct_query_template, step_by_step_template,
                    direct_query_template_cls, step_by_step_template_cls)

class DecompDataset(Dataset):
    def __init__(self, decomp_inputs, tokenizer, max_words=100, pad=True):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.decomp_items = decomp_inputs
    
    def __len__(self):
        return len(self.decomp_items)
    
    def pad_token(self, input_id):
        if self.pad:
            padding = self.max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: self.max_words]
        return input_id
    
    def __getitem__(self, index):
        examples = []
        labels = []
        example_masks = []
        for i in index:
            decomp_item = self.decomp_items[i]
            query = decomp_item["question"]
            decomp_list = decomp_item["decomposition"]
            # obtain the length of prefix id
            prefix_id = torch.tensor(
                self.tokenizer.encode(query), dtype=torch.int64
            )
            
            # create input ids
            decomp_str = f" {self.tokenizer.bos_token} ".join(decomp_list)
            # input_str = query + self.tokenizer.bos_token + decomp_str
            input_id = torch.tensor(
                self.tokenizer.encode(query), dtype=torch.int64
            )
            if self.pad:
                input_id = self.pad_token(input_id)
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            att_mask = att_mask.float()

            # create target ids
            label_id = torch.tensor(
                self.tokenizer.encode(decomp_str), dtype=torch.int64
            )
            if self.pad:
                label_id = self.pad_token(label_id)
            label_mask = label_id.ge(0)
            label_id[~label_mask] = IGNORE_INDEX

            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
        }

class ReplaceDataset(Dataset):
    def __init__(self, replace_inputs, tokenizer, max_words=100, pad=True):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.replace_items = replace_inputs
    
    def __len__(self):
        return len(self.replace_items)
    
    def pad_token(self, input_id):
        if self.pad:
            padding = self.max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: self.max_words]
        return input_id
    
    def __getitem__(self, index):
        examples = []
        labels = []
        example_masks = []
        for i in index:
            replace_item = self.replace_items[i]
            raw_query = replace_item["raw query"].strip()
            attrs = replace_item["attributes"]
            rpl_query = replace_item["replaced query"].strip()
            # obtain the length of prefix id
            prompt = f"Attributes {self.tokenizer.bos_token} "
            for attr in attrs:
                prompt += attr + f" {self.tokenizer.bos_token} "
            prompt += f"Sentence {self.tokenizer.bos_token} {raw_query}"
            
            # create input ids
            input_id = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            input_id = self.pad_token(input_id)
            
            att_mask = input_id.ge(0)
            input_id[~att_mask] = 0
            att_mask = att_mask.float()

            # create target ids
            label_id = torch.tensor(
                self.tokenizer.encode(rpl_query), dtype=torch.int64
            )
            label_id = self.pad_token(label_id)
            label_mask = label_id.ge(0)
            label_id[~label_mask] = IGNORE_INDEX

            examples.append(input_id)
            labels.append(label_id)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
        }

class EvalReplace(Dataset):
    def __init__(self, replace_inputs, tokenizer, max_words=100, pad=True):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.replace_items = replace_inputs
    
    def __len__(self):
        return len(self.replace_items)
    
    def pad_token(self, input_id):
        if self.pad:
            padding = self.max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: self.max_words]
        return input_id
    
    def __getitem__(self, index):
        examples = []
        example_masks = []
        indexes = []
        questions = []
        for i in index:
            replace_item = self.replace_items[i]
            raw_query = replace_item["raw query"].strip()
            attrs = replace_item["attributes"]
            idx = replace_item["index"]
            question = replace_item["question"]
            # obtain the length of prefix id
            prompt = f"Attributes {self.tokenizer.bos_token} "
            for attr in attrs:
                prompt += attr + f" {self.tokenizer.bos_token} "
            prompt += f"Sentence {self.tokenizer.bos_token} {raw_query}"
            
            # create input ids
            input_id = torch.tensor(
                self.tokenizer.encode(prompt), dtype=torch.int64
            )
            input_id = self.pad_token(input_id)
            
            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()

            examples.append(input_id)
            example_masks.append(att_mask)
            indexes.append(idx)
            questions.append(question)

        return {
            "input_ids": examples,
            "attention_mask": example_masks,
            "index": indexes,
            "question": questions,
        }

class FluentDataset(Dataset):
    def __init__(self, fluent_inputs, tokenizer, max_words=100, pad=True):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.fluent_inputs = fluent_inputs
    
    def __len__(self):
        return len(self.fluent_inputs)
    
    def pad_token(self, input_id):
        if self.pad:
            padding = self.max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: self.max_words]
        return input_id
    
    def __getitem__(self, index):
        examples = []
        labels = []
        example_masks = []
        for i in index:
            fluent_item = self.fluent_inputs[i]
            sentence = fluent_item["sentence"].strip()
            label = fluent_item["score"]
            
            # create input ids
            input_id = torch.tensor(
                self.tokenizer.encode(sentence), dtype=torch.int64
            )
            input_id = self.pad_token(input_id)
            
            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()

            examples.append(input_id)
            labels.append(label)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
        }

class RecompDataset(Dataset):
    def __init__(self, inputs, tokenizer, max_words=500, pad=True, args=None, classification=True, causal=False, test=False):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.inputs = inputs
        self.args = args
        self.cls = classification
        self.causal = causal
        self.test = test
    
    def __len__(self):
        return len(self.inputs)
    
    def pad_token(self, input_id, max_words):
        if self.pad:
            padding = max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: max_words]
        if input_id[-1].item() != -1:
            input_id[-1] = self.tokenizer.eos_token_id
        return input_id
    
    def left_pad_token(self, input_id, max_words):
        if self.pad:
            padding = max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((torch.zeros(padding, dtype=torch.int64) - 1, input_id))
            elif padding < 0:
                input_id = input_id[: max_words]
        return input_id
    
    def __getitem__(self, index):
        # IGNORE_INDEX = self.tokenizer.pad_token_id
        examples = []
        labels = []
        example_masks = []
        questions = []
        for i in index:
            item = self.inputs[i]
            query = item["question"]
            decompositions = item["decomposition"][:-1]
            final_question = item["decomposition"][-1]
            subanswers = item["sub-answers"][:len(decompositions)]
            label = item["label"]
            
            # create question ids
            query_id = torch.tensor(
                self.tokenizer.encode(query), dtype=torch.int64
            )
            query_id = self.pad_token(query_id, self.max_words)
            query_mask = query_id.ge(0)
            query_id[~query_mask] = self.tokenizer.pad_token_id

            # create prompt
            prompt = ""
            prompt = prompt + query
            for cnt, (decomp, subans) in enumerate(zip(decompositions, subanswers), start=1):
                prompt += f"{self.tokenizer.bos_token} #{cnt}: {decomp} {subans}"
            prompt += f" {self.tokenizer.bos_token} {final_question}"

            
            if self.causal and (not self.cls) and (not self.test):
                example = prompt + " " + label
                input_id = torch.tensor(
                            self.tokenizer.encode(example), dtype=torch.int64
                        )
            else:
                # create input ids
                input_id = torch.tensor(
                    self.tokenizer.encode(prompt), dtype=torch.int64
                )
            if self.test and (not self.cls) and self.causal:
                input_id = self.left_pad_token(input_id, self.max_words)
            else:
                input_id = self.pad_token(input_id, self.max_words)

            # create target ids
            if self.cls:
                labels.append(label)
            else:
                if self.causal and (not self.test):
                    prompt_id = torch.tensor(
                        self.tokenizer.encode(prompt), dtype=torch.int64
                    )
                    # print(prompt_id)
                    label_id = copy.deepcopy(input_id)
                    # print(label_id)
                    label_id[:(len(prompt_id)-1)] = -1
                    label_mask = label_id.ge(0)
                else:
                    label_id = torch.tensor(
                        self.tokenizer.encode(label), dtype=torch.int64
                    )
                    if self.pad:
                        label_id = self.pad_token(label_id, 50)
                    label_mask = label_id.ge(0)
                label_id[~label_mask] = IGNORE_INDEX
                labels.append(label_id)

            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()

            examples.append(input_id)
            example_masks.append(att_mask)
            questions.append(query_id)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
            "question": questions,
        }

class RecompPTDataset(Dataset):
    # dataset to pretrain the recomposor
    def __init__(self, inputs, tokenizer, max_words=512, pad=True, args=None, classification=True, causal=False):
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad
        self.inputs = inputs
        self.args = args
        self.cls = classification
        self.causal = causal
    
    def __len__(self):
        return len(self.inputs)
    
    def pad_token(self, input_id, max_words):
        if self.pad:
            padding = max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((input_id, torch.zeros(padding, dtype=torch.int64) - 1))
            elif padding < 0:
                input_id = input_id[: max_words]
        if input_id[-1].item() != -1:
            input_id[-1] = self.tokenizer.eos_token_id
        return input_id

    def left_pad_token(self, input_id, max_words):
        if self.pad:
            padding = max_words - input_id.shape[0]
            if padding > 0:
                input_id = torch.cat((torch.zeros(padding, dtype=torch.int64) - 1, input_id))
            elif padding < 0:
                input_id = input_id[: max_words]
        return input_id
    
    def encode_input(self, context, query):
        if len(context) > 0:
            input_id = torch.tensor(
                self.tokenizer.encode(context, query), dtype=torch.int64
            )
        else:
            input_id = torch.tensor(
                self.tokenizer.encode(query), dtype=torch.int64
            )
        return input_id

    def __getitem__(self, index):
        # IGNORE_INDEX = self.tokenizer.pad_token_id
        examples = []
        labels = []
        example_masks = []
        for i in index:
            item = self.inputs[i]
            query = item["question"]
            context = item["context"]
            label = item["label"]
            
            # create input ids
            if self.causal and (not self.cls):
                example = query + " " + label
            else:
                # create input ids
                example = query
            input_id = self.encode_input(context, query)
            
            input_id = self.pad_token(input_id, self.max_words)
            
            att_mask = input_id.ge(0)
            input_id[~att_mask] = self.tokenizer.pad_token_id
            att_mask = att_mask.float()

            # create target ids           
            if self.cls:
                labels.append(label)
            else:
                if self.causal:
                    prompt_id = torch.tensor(
                        self.tokenizer.encode(query), dtype=torch.int64
                    )
                    # print(prompt_id)
                    label_id = copy.deepcopy(input_id)
                    # print(label_id)
                    label_id[:(len(prompt_id)-1)] = -1
                else:
                    label_id = torch.tensor(
                        self.tokenizer.encode(label), dtype=torch.int64
                    )
                    if self.pad:
                        label_id = self.pad_token(label_id, 50)
                label_mask = label_id.ge(0)
                label_id[~label_mask] = IGNORE_INDEX
                labels.append(label_id)

            examples.append(input_id)
            example_masks.append(att_mask)

        return {
            "input_ids": examples,
            "labels": labels,
            "attention_mask": example_masks,
        }

class BslData(Dataset):
    def __init__(self, inputs, args=None):
        self.inputs = inputs
        self.args = args
        data_type = dataset_type[args.eval_data]
        self.data_type = data_type
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        prompts = []
        labels = []
        questions = []
        for i in index:
            item = self.inputs[i]
            question = item["question"]
            questions.append(question)
            if self.args.step_by_step:
                template = step_by_step_template if self.data_type=="qa" else step_by_step_template_cls
                prompt = template.format(question=question)
            else:
                template = direct_query_template if self.data_type=="qa" else direct_query_template_cls
                prompt = template.format(question=question)
            prompt = chat_templates[self.args.eval_model].format(prompt=prompt)
            prompts.append(prompt)
            label = item["label"]
            labels.append(label)
        return {"sequences": prompts, 'labels': labels, "questions": questions}
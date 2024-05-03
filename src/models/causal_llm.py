from transformers import (AutoTokenizer, 
                        AutoModelForCausalLM,
                        GenerationConfig)
from huggingface_hub import login
login(token="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")

class SpicyBoros:
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        self.generation_config = GenerationConfig(
                **kwargs
            )
    
    def process_responses(self, responses, prompts):
        outputs = []
        for prompt, response in zip(prompts, responses):
            response = response.replace(prompt, "")
            outputs.append(response.strip())
        return outputs

    def generate(self, prompts):
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt")
        output_ids = self.model.generate(
                        input_ids=inputs["input_ids"].to(self.model.device),
                        attention_mask=inputs["attention_mask"].to(self.model.device),
                        generation_config = self.generation_config,)
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        responses = self.process_responses(responses, prompts)
        return responses

class Vicuna(SpicyBoros):
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        self.generation_config = GenerationConfig(
                **kwargs
            )

class YiChat(SpicyBoros):
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        self.generation_config = GenerationConfig(
                **kwargs
            )

class LLaMa(SpicyBoros):
    def __init__(self, model_name, **kwargs):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto").eval()
        self.generation_config = GenerationConfig(
                **kwargs
            )
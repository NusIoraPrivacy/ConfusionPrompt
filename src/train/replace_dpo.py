from dataclasses import dataclass, field
from typing import Optional
from utils.param import parse_args
from utils.utils import get_model_tokenizer

import json
from datasets import Dataset
from peft import LoraConfig
from transformers import  HfArgumentParser, TrainingArguments, get_scheduler
from torch import optim

from trl import DPOTrainer

from argparse import Namespace

@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    model_name_or_path: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "optimizer learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    max_length: Optional[int] = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: Optional[int] = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: Optional[int] = field(
        default=512, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    label_pad_token_id: Optional[int] = field(default=-100, metadata={"help": "label for non response tokens"})
    max_steps: Optional[int] = field(default=1000, metadata={"help": "max number of training steps"})
    epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs"})
    # lora parameters
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    # instrumentation
    sanity_check: Optional[bool] = field(default=True, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )
    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )

def merge_args(args1: Namespace, args2: Namespace) -> Namespace:
    """
    Merges two namespaces but throws an error if there are keys that collide.

    ref: https://stackoverflow.com/questions/56136549/how-can-i-merge-two-argparse-namespaces-in-python-2-x
    :param args1:
    :param args2:
    :return:
    """
    # - the merged args
    # The vars() function returns the __dict__ attribute to values of the given object e.g {field:value}.
    args = Namespace(**vars(args1), **vars(args2))
    return args

def create_prompt_rpl(raw_query, attrs, tokenizer):
    prompt = f"Attributes {tokenizer.bos_token} "
    for attr in attrs:
        prompt += attr + f" {tokenizer.bos_token} "
    prompt += f"Sentence {tokenizer.bos_token} {raw_query.strip()}"
    return prompt

def get_data(args, tokenizer):
    """Load the data with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }
    """
    data_path = f"{args.root_path}/results/{args.decomp_data}/replace/test_samples_final.json"
    dataset_dict = {"prompt":[], "chosen":[], "rejected":[]}
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            raw_query, attrs, replacements, ranks = this_item["raw query"], this_item["attributes"], this_item["replacement"], this_item["rank"]
            prompt = create_prompt_rpl(raw_query, attrs, tokenizer)
            if len(replacements) <= 1:
                continue
            for i in range(len(replacements)-1):
                this_r = ranks[i]
                for j in range(i+1, len(replacements)):
                    next_r = ranks[j]
                    if next_r < this_r:
                        dataset_dict['prompt'].append(prompt)
                        dataset_dict["chosen"].append(replacements[j].strip())
                        dataset_dict["rejected"].append(replacements[i].strip())
                    elif this_r < next_r:
                        dataset_dict['prompt'].append(prompt)
                        dataset_dict["chosen"].append(replacements[i].strip())
                        dataset_dict["rejected"].append(replacements[j].strip())
    return Dataset.from_dict(dataset_dict)

if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    args = parse_args()
    model_name = args.base_model.split("/")[-1]

    # 1. load a pretrained model and tokenizer
    model_dir = f"{args.root_path}/save_models/replace/{args.decomp_data}/{model_name}/final"
    tokenizer, model = get_model_tokenizer(model_dir, args)
    _, model_ref = get_model_tokenizer(model_dir, args)

    # 2. Load the Anthropic Helpful-Harmless dataset
    train_dataset = get_data(args, tokenizer)

    # # 3. Load evaluation dataset
    # eval_dataset = get_data(args)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        # max_steps=script_args.max_steps,
        max_steps=-1,
        save_strategy = "epoch",
        num_train_epochs = script_args.epochs, 
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="no",
        # logging_first_step=True,
        logging_steps=10,  # match results in blog post
        # eval_steps=500,
        output_dir=f"{args.root_path}/save_models/dpo/replace/{model_name}",
        # optim="rmsprop",
        # warmup_steps=150,
        report_to=script_args.report_to,
        # bf16=True,
        gradient_checkpointing=script_args.gradient_checkpointing,
    )

    if script_args.use_peft:
        peft_config = LoraConfig(
            r=script_args.peft_lora_r,
            lora_alpha=script_args.peft_lora_alpha,
            bias="none",
            task_type="CAUSAL_LM",
        )
    else:
        peft_config = None

    optimizer = optim.AdamW(
                    model.parameters(),
                    lr=script_args.learning_rate,
                    # weight_decay=0.0,
                )
    num_training_steps = script_args.epochs * int(len(train_dataset)/script_args.per_device_train_batch_size)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
        )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=False,
        peft_config=peft_config,
        # optimizers=(optimizer, lr_scheduler),
    )

    # 6. train
    dpo_trainer.train()
    model_dir = f"{args.root_path}/save_models/dpo/replace/{model_name}/final"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
from datasets import load_dataset
import json
import pandas as pd
from utils.globals import mmlu_decomp_template

cat_dict = {"business": ["business_ethics", "marketing"],
            "legal": ["international_law", "jurisprudence"],
            "politics": ["us_foreign_policy", "high_school_government_and_politics"],
            "medicine": ["college_medicine", "clinical_knowledge", "nutrition"],
            "religion": ["world_religions"]
            }

def creat_mmlu_dict(data_subset, cat):
    out = []
    questions = data_subset["question"]
    choices = data_subset["choices"]
    answers = data_subset["answer"]
    for question, choice, answer in zip(questions, choices, answers):
        this_dict = {
            "question": question,
            "choices": choice,
            "answer": answer,
            "category": cat,
        }
        out.append(this_dict)

    return out

def creat_mmlu_cat_data(root):
    filter_data = []
    for cat in cat_dict:
        sub_cats = cat_dict[cat]
        for sub_cat in sub_cats:
            data_subset = load_dataset("cais/mmlu", sub_cat, split="test")
            filter_data.extend(creat_mmlu_dict(data_subset, cat))
            data_subset = load_dataset("cais/mmlu", sub_cat, split="validation")
            filter_data.extend(creat_mmlu_dict(data_subset, cat))

    with open(f'{root}/data/mmlu/raw.json', 'w') as fout:
        json.dump(filter_data, fout, indent=4)


    df = pd.DataFrame.from_records(filter_data)
    df.to_csv(f"{root}/data/mmlu/raw.csv", index=False)

def create_prompt_template(demos, category, template, target_question, target_answer):
    related_data = []
    for sample in demos:
        if sample["category"] == category:
            related_data.append(sample)
    format_strings = []
    for sample in related_data:
        format_strings.append(sample["question"])
        answer = sample["choices"][sample["answer"]]
        format_strings.append(answer)
        decomp_list = sample["decomposition"]
        decomp_str = "["
        for i, decomp in enumerate(decomp_list):
            if i > 0:
                decomp_str += " "
            decomp_str += ('"' + decomp + '",')
        decomp_str += "]"
        format_strings.append(decomp_str)
    format_strings.extend([target_question, target_answer])
    template = template.format(*format_strings)
    return template

def gen_decomp_samples_gpt(root):
    with open(f"{root}/data/mmlu/raw_decom.json") as f:
        decomp_demo = json.load(f)
    with open(f"{root}/data/mmlu/raw.json") as f:
        raw_data = json.load(f)
    prompts = []
    for sample in raw_data:
        cat = sample["category"]
        question = sample["question"]
        answer = sample["choices"][sample["answer"]]
        prompt = create_prompt_template(decomp_demo, cat, mmlu_decomp_template, question, answer)
        prompts.append(prompt)
    return prompts

if __name__ == "__main__":
    prompts = gen_decomp_samples_gpt("/opt/data/private/ConfusionPrompt")
    print(len(prompts))
    print(prompts[-2])
    print(prompts[-1])

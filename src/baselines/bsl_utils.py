from utils.globals import *

def seq_collate_fn(seq_list):
    output = {key:[] for key in seq_list[0]}
    for seq_dict in seq_list:
        for key in seq_dict:
            output[key].append(seq_dict[key])
    return output

def standard_ans(ans):
    ans = ans.lower()
    ans = ans.strip(".")
    ans = ans.strip()
    pred = 1 if "yes" in ans else 0
    return pred

def create_query_prompt(questions):
    prompts = []
    for question in questions:
        prompt = cls_bool_template.format(question=question)
        prompts.append(prompt)
    return prompts
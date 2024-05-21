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

def create_query_prompt(questions, contexts, args):
    data_type = dataset_type[args.eval_data]
    prompts = []
    for question, context in zip(questions, contexts):
        template = direct_query_template if data_type=="qa" else direct_query_template_cls
        prompt = template.format(question=question)
        if args.use_context:
            context = " ".join(context)
            prompt = context_template.format(context=context) + prompt
        prompts.append(prompt)
    return prompts
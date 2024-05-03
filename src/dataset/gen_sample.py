from utils.utils import write_list, get_response
from utils.param import parse_args
from utils.globals import *
import json
from openai import OpenAI
import itertools
from tqdm import tqdm
from utils.utils import *
from models.key import _API_KEY

def create_message_replace(attr_list, n_replaces, questions):
    template = replace_template_multiple
    # convert attribute list/tuple to string
    attr_string = ""
    for i, attr in enumerate(attr_list):
        if i == 0:
            attr_string += f"\"{attr}\""
        elif i == len(attr_list) - 1:
            attr_string += f" and \"{attr}\""
        else:
            attr_string += f", \"{attr}\""
    query_dict = {}
    # convert question list/tuple to string
    for cnt, question in enumerate(questions, 1):
        query_dict[f"S{cnt}"] = question
    query_str = json.dumps(query_dict)
    prompt = template.format(attributes=attr_string, n_replaces=n_replaces, sentences=query_str)
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages, query_dict

DATA_NAME = "strategyQA"
N_REPLACES = 3

if __name__ == '__main__':
    args = parse_args()
    args.decomp_data = DATA_NAME
    args.n_replaces = N_REPLACES
    data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/test_demo_rank_100_200.json"
    output_path = f"{args.root_path}/results/{args.decomp_data}/replace/train_samples_multiple_100_200.json"
    dataset = []
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            dataset.append(this_item)

    client = OpenAI(
        api_key=_API_KEY,
    )

    outputs = []
    for item in tqdm(dataset):
        attrs, question, qualified_decomp, attr2decomps, decompositions = item["private attributes"], item["question"], item["qualified decomp"], item["has_attrs"], item["decomposition"]
        attr_combs = []
        for r in range(1, len(attrs) + 1):
            attr_combs.extend(list(itertools.combinations(attrs, r)))
        # create a decomp2attr dictionary for all qualified decompostions
        decomp2attrs = {}
        qualified_decomp = process_candidate(qualified_decomp)
        for idx in qualified_decomp:
            this_decomp = decompositions[idx]
            attr2decomp = attr2decomps[idx]
            decomp_list = decomp2list(this_decomp, args)
            # create a dictionary for decomposition to attribute
            decomp2attrs = {decomp:[] for decomp in decomp_list} # {sub-question: [attributes it contains]}
            for attr in attr2decomp:
                decomp_string = attr2decomp[attr]
                decomp_idx = process_has_attr(decomp_string)
                for idx in decomp_idx:
                    decomp2attrs[decomp_list[idx-1]].append(attr)
        for attr_comb in attr_combs:
            attr_comb = list(attr_comb)
            questions = [question]
            for decomp, this_attrs in decomp2attrs.items():
                if set(attr_comb).issubset(this_attrs):
                    questions.append(decomp)
            messages, query_dict = create_message_replace(attr_comb, args.n_replaces, questions)
            result = get_response(client, messages, args)
            outputs.append({
                "raw query": query_dict,
                "attributes": attr_comb,
                "replaced query": result
            })
            write_list(output_path, outputs)
    
    # convert the dataset to a format with each line consisting of only one sentence/question
    data_path = f"{args.root_path}/results/{args.decomp_data}/replace/train_samples_multiple_100_200.json"
    output_path = f"{args.root_path}/results/{args.decomp_data}/replace/train_samples_100_200.json"
    dataset = []
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            dataset.append(this_item)
    outputs = []
    for item in dataset:
        questions, attributes, replace_query = item["raw query"], item["attributes"], item["replaced query"]
        replace_query = eval(replace_query)
        for key in questions:
            this_query = questions[key]
            replace_list = replace_query[key]
            for replace in replace_list:
                outputs.append({
                    "raw query": this_query,
                    "attributes": attributes,
                    "replaced query": replace
                })
        write_list(output_path, outputs)
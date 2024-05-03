from utils.param import parse_args
from utils.utils import *
from utils.globals import *

import json
from openai import (
    OpenAI,
)
from tqdm import tqdm
import re
from nltk.stem import PorterStemmer
from scipy.stats import rankdata
from models.key import _API_KEY

# pst = PorterStemmer()
def get_stem(attr, stemmer):
    if len(attr.split(" ")) > 1:
        return attr
    else:
        attr = stemmer.stem(attr)
        return attr

def create_message_simple(decomposition):
    template = hardness_template2
    prompt = template.format(decomposition = decomposition)
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages

def create_message_attribute(attribute, decomp_list):
    template = has_attr_template
    cnt = 0
    decomp_str = ""
    for decomp in decomp_list:
        cnt += 1
        decomp_str += f"{cnt}. {decomp}"
    
    prompt = template.format(attribute = attribute, sentences = decomp_str)
    messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    return messages

if __name__ == '__main__':
    args = parse_args()
    # load datas
    # data_path = f"{args.root_path}/results/strategyQA/decomp/test_demo.json"
    data_path = f"{args.root_path}/results/strategyQA/decomp/test_demo_rank_400_600.json"  # for debugging purpose
    output_path = f"{args.root_path}/results/strategyQA/decomp/test_demo_rank_400_600.json"
    dataset = []
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            dataset.append(this_item)

    client = OpenAI(
        api_key=_API_KEY,
    )

    # ### evaluate decomposition quality
    # outputs = []
    # for item in tqdm(dataset):
    #     question, attrs, decompositions = item["question"], item["private attributes"], item["decomposition"]
    #     # obtain qualified decompositions
    #     messages = create_message_qualify(question, decompositions)
    #     result = get_response(client, messages, args)
    #     item["qualified decomp"] = result
    #     # evaluate hardness of the last question, to judge if it's simple enough for the local re-composition model to answer
    #     is_simple = []
    #     for decompostion in decompositions:
    #         messages = create_message_simple(decompositions)
    #         result = get_response(client, messages, args)
    #         results = re.search(r"#theclass:((.|\n)+)#thereason", result)
    #         if results is not None:
    #             results = list(results.groups())
    #         else:
    #             results = []
    #         results2 = re.search(r"#1theclass:((.|\n)+)#1thereason", result)
    #         if results2 is not None:
    #             results2 = list(results2.groups())
    #         else:
    #             results2 = []
    #         results = results + results2
    #         if len(results) > 0:
    #             number_results= re.findall(r'(\d+)', results[0])
    #             if len(number_results) > 0:
    #                 is_simple.append(number_results[0])
    #             else:
    #                 is_simple.append(results[0])
    #         else:
    #             is_simple.append(result)
    #     item["simple last"] = is_simple
    #     outputs.append(item)
    #     write_list(output_path, outputs)
    
    # ### compute complexity
    # # dataset = outputs
    # outputs = []
    # # query GPT to identify if each sub question directly contain private attribute
    # for item in tqdm(dataset):
    #     attrs, decompositions = item["private attributes"], item["decomposition"]
    #     has_attrs_list = []
    #     for decomposition in decompositions:
    #         # convert decompositons into list of sub-questions
    #         decomp_list = decomp2list(decomposition, args)
    #         # identify if each sub question directly contain private attribute
    #         attrs2decomp = {} # {attr:[index of subquestions containing the attribute]}
    #         for attr in attrs:
    #             messages = create_message_attribute(attr, decomp_list)
    #             result = get_response(client, messages, args)
    #             attrs2decomp[attr] = result
    #         has_attrs_list.append(attrs2decomp)
    #     item["has_attrs"] = has_attrs_list
    #     outputs.append(item)
    #     write_list(output_path, outputs)

    # dataset = outputs
    outputs = []
    complexities = []
    # compute complexity based on the number of attributes
    for item in dataset:
        attrs, has_attrs_list, decompositions, simple_lasts = item["private attributes"], item["has_attrs"], item["decomposition"], item["simple last"]
        complexities = []
        for attrs2decomp, decomposition, simple_last in zip(has_attrs_list, decompositions, simple_lasts):
            # convert decompositons into list of sub-questions
            decomp_list = decomp2list(decomposition, args, simple_last)
            # create a dictionary for decomposition to attribute
            decomp2attrs = {idx:[] for idx in range(len(decomp_list))} # {index of decomp: [attributes it contains]}
            for attr in attrs2decomp:
                decomp_string = attrs2decomp[attr]
                decomp_idx = process_has_attr(decomp_string)
                for idx in decomp_idx:
                    if idx <= len(decomp_list):
                        decomp2attrs[idx-1].append(attr)
            # compute the complexity for each query
            complexity = 0
            for i, single_query in enumerate(decomp_list):
                # if the query contain answers from previous question like #1 or #2, 
                # find the related attributes in previous question, and add the attribute
                indexes = re.findall(r'#(\d+)', single_query)
                for idx in indexes:
                    idx = int(idx)
                    # ensure the validity of the index
                    if idx <= len(decomp_list):
                        related_query_attrs = decomp2attrs[idx-1]
                        decomp2attrs[i] = decomp2attrs[i] + related_query_attrs
                attr_cnt = len(set(decomp2attrs[i]))
                if attr_cnt > 0:
                    complexity += (1/args.mu) ** attr_cnt
            complexities.append(complexity)
        item["complexity"] = complexities
        outputs.append(item)
    # output the response
    write_list(output_path, outputs)

    ### compute final ranking
    dataset = outputs
    outputs = []
    for item in dataset:
        decompositions, qualify_cand, complexities = item["decomposition"], item["qualified decomp"], item["complexity"]
        all_ranks = [0] * len(decompositions)
        # first obtain the qualified decompositions, it should be rank before the unqualified ones
        qualify_cand = process_candidate(qualify_cand)
        qualify_complex = []
        for idx in qualify_cand:
            qualify_complex.append(complexities[idx])
        complex_rank = list(rankdata(qualify_complex, method='min'))

        for idx, rank in zip(qualify_cand, complex_rank):
            all_ranks[idx] = int(rank)
        
        # then rank the unqualified decompositions
        unqualify_cand = []
        for i in range(len(all_ranks)):
            if all_ranks[i] == 0:
                unqualify_cand.append(i)
        
        unqualify_complex = []
        for idx in unqualify_cand:
            unqualify_complex.append(complexities[idx])
        complex_rank = list(rankdata(unqualify_complex, method='min'))

        for idx, rank in zip(unqualify_cand, complex_rank):
            all_ranks[idx] = int(rank + len(qualify_cand))

        item["rank"] = all_ranks
        outputs.append(item)
    write_list(output_path, outputs)
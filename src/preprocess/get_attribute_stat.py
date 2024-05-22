import random
from utils.utils import read_data, write_list
import os
import numpy as np
import json

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def get_attribute_stat(query2attr):
    attr_num = []
    attr_length = []
    attr_pct = []
    for query, attrs in query2attr.items():
        attr_num.append(len(attrs))
        this_attr_len = 0
        for attr in attrs:
            this_len = len(attr.split())
            attr_length.append(this_len)
            this_attr_len += this_len
        query_len = len(query.split())
        pct = this_attr_len/query_len
        attr_pct.append(pct)
    print("Max number:", max(attr_num))
    print("Min number:", min(attr_num))
    print("Average number:", np.mean(attr_num))
    print("Max length:", max(attr_length))
    print("Min length:", min(attr_length))
    print("Average length:", np.mean(attr_length))
    print("Average percentage:", np.mean(attr_pct))

# strategyQA
data_path = f"{_ROOT_PATH}/results/strategyQA/replace/question_attrs.json"
with open(data_path) as f:
    query2attr = json.load(f)
print("Attribute statistics for strategyQA")
get_attribute_stat(query2attr)

# MuSiQue
data_path = f"{_ROOT_PATH}/results/musique/replace/question_attrs.json"
with open(data_path) as f:
    query2attr = json.load(f)
print("Attribute statistics for MuSiQue")
get_attribute_stat(query2attr)

# P2F
data_path = f"{_ROOT_PATH}/data/p2f/question_attrs_gpt-4o.json"
with open(data_path) as f:
    query2attr = json.load(f)
print("Attribute statistics for P2F")
get_attribute_stat(query2attr)
import random
from utils.utils import read_data, write_list
data_path = "/home/sixing/llm/LLM_decompose/results/strategyQA/replace/fluency_trainset.json"
dataset = read_data(data_path)
num_to_select = 100
list_of_random_items = random.sample(dataset, num_to_select)
output_path = "/home/sixing/llm/LLM_decompose/results/strategyQA/replace/sample_fluency.json"
write_list(output_path, list_of_random_items)
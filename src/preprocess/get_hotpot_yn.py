import json
import os
from utils.utils import read_data

if __name__ == "__main__":
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(parent_dir))
    ### generate training set for y/n question with decomposition
    train_data = []
    data_path = f"{root_dir}/data/hotpotqa/train_decomp.json"
    with open(data_path) as f:
        dataset = json.load(f)
    
    for sample in dataset:
        question, answer, decompositions = sample["question"], sample["answer"], sample["decompositions"]
        decomposition = [i["question"] for i in decompositions]
        subanswer = [i["answer"] for i in decompositions]
        if answer in ["yes", "no"]:
            item = {"question": question, "answer": answer, "sub-answer": subanswer, "decomposition": decomposition}
            train_data.append(item)
    

    data_path = f"{root_dir}/data/hotpotqa/train_attrs_decomp_gpt-4_ans_final.json"
    dataset = read_data(data_path)

    for sample in dataset:
        question, answer, decomposition, subanswer, f1 = sample["question"], sample["label"], sample["decomposition"], sample["subanswers"], sample["f1"]
        if answer in ["yes", "no"] and f1 > 0.6:
            item = {"question": question, "answer": answer, "sub-answer": subanswer, "decomposition": decomposition}
            train_data.append(item)
    print(len(train_data))
    output_path = f"{root_dir}/data/hotpotqa-yn/train_decomp_yn.json"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(train_data, f)

    ### extract dataset from test set
    test_data = []
    data_path = f"{root_dir}/data/hotpotqa/hotpot_dev_distractor_v1.json"
    with open(data_path) as f:
        dataset = json.load(f)
    for sample in dataset:
        question, answer = sample["question"], sample["answer"]
        if answer in ["yes", "no"]:
            item = {"question": question, "answer": answer}
            test_data.append(item)
    print(len(test_data))
    output_path = f"{root_dir}/data/hotpotqa-yn/test_yn.json"
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(test_data, f)
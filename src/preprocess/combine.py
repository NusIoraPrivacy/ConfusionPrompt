import json
from utils.param import parse_args
from utils.utils import write_list

args = parse_args()
# dataset = {"spacy":[], "flair":[], "bert":[]}
dataset = {"spacy":[], "textblob":[]}

for data_name in ["spacy", "textblob"]:
    data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/ner/decompose_{data_name}.json"
    with open(data_path) as dataset_file:
        for line in dataset_file:
            this_item = json.loads(line)
            dataset[data_name].append(this_item)

n_samples = len(dataset["spacy"])

outputs = []
for i in range(n_samples):
    priv_attrs = []
    for data_name in ["spacy", "textblob"]:
        this_item = dataset[data_name][i]
        priv_attrs += this_item["private attributes"]
    priv_attrs = list(set(priv_attrs))

    question = dataset["spacy"][i]["question"]

    decompositions = dataset["spacy"][i]["decomposition"]

    outputs.append({"question":question, 
                   "private attributes":priv_attrs, 
                   "decomposition": decompositions})

output_path = f"{args.root_path}/results/{args.decomp_data}/ner/decompose_combined.json"
write_list(output_path, outputs)
import spacy
from utils.utils import write_list, read_data, sing2multi
from utils.param import parse_args
import json
import os
from tqdm import tqdm
# from flair.data import Sentence
# from flair.nn import Classifier
import re
from transformers import pipeline
from textblob import TextBlob

def extract_ents_spacy(nlp, sent): 
    doc = nlp(sent) 
    output = []
    if doc.ents: 
        for ent in doc.ents: 
            output.append(ent.text)
    return output

def extract_ents_flair(sent):
    sentence = Sentence(sent)
    tagger = Classifier.load('ner')
    tagger.predict(sentence)
    entitys = []
    for label in sentence.labels:
        label = label.unlabeled_identifier
        entity = re.findall(r'\"(.*?)\"', label)
        if len(entity) > 0:
            entitys.append(entity[0])
    return entitys

def extract_ents_bert(ner_tagger, sent): 
    outputs = ner_tagger(sent)
    entitys = []
    for item in outputs:
        if item['score'] >= 0.6:
            entitys.append(item['word'])
    return entitys

def extract_noun_phrases(sentence):
    blob = TextBlob(sentence)
    return blob.noun_phrases

if __name__ == '__main__':
    args = parse_args()
    # read data
    data_path = f"{args.root_path}/results/{args.decomp_data}/decomp/bart_pretrained/decompose_sp_test_epoch_1.json"
    dataset = read_data(data_path)
    dataset = sing2multi(["question"], ["decomposition"], dataset)

    # # extract entities/private attributes
    # outputs = []
    # if args.ent_algo == "spacy":
    #     nlp = spacy.load('en_core_web_sm')
    # if args.ent_algo == "bert":
    #     ner_tagger = pipeline("ner", aggregation_strategy="simple", 
    #                     model="dbmdz/bert-large-cased-finetuned-conll03-english",
    #                     device=1,
    #                     # device_map="auto",
    #                     )
    # for i in tqdm(range(0, len(dataset), args.n_resp)):
    #     question = dataset[i]["question"]
    #     if args.ent_algo == "spacy":
    #         entitys = extract_ents_spacy(nlp, question)
    #     elif args.ent_algo == "flair":
    #         entitys = extract_ents_flair(question)
    #     elif args.ent_algo == "bert":
    #         entitys = extract_ents_bert(ner_tagger, question)
    #     elif args.ent_algo == "textblob":
    #         entitys = extract_noun_phrases(question)
    #     decompositions = []
    #     for j in range(args.n_resp):
    #         this_decomp = dataset[i+j]["decomposition"]
    #         decompositions.append(this_decomp)
    #     this_item = {"question": question, "decomposition": decompositions, "private attributes": entitys}
    #     outputs.append(this_item)

    #     # save results
    #     output_path = f"{args.root_path}/results/{args.decomp_data}/decomp/ner/decompose_{args.ent_algo}.json"
    #     write_list(output_path, outputs)
    
    args.ent_algo = "bert"
    outputs = []
    if args.ent_algo == "spacy":
        nlp = spacy.load('en_core_web_sm')
    if args.ent_algo == "bert":
        ner_tagger = pipeline("ner", aggregation_strategy="simple", 
                        model="dbmdz/bert-large-cased-finetuned-conll03-english",
                        device=1,
                        # device_map="auto",
                        )
    for item in tqdm(dataset):
        question = item["question"]
        decompositions = item["decomposition"]
        if args.ent_algo == "spacy":
            entitys = extract_ents_spacy(nlp, question)
        elif args.ent_algo == "flair":
            entitys = extract_ents_flair(question)
        elif args.ent_algo == "bert":
            entitys = extract_ents_bert(ner_tagger, question)
        elif args.ent_algo == "textblob":
            entitys = extract_noun_phrases(question)
        this_item = {"question": question, "private attributes": entitys, "decomposition": decompositions}
        outputs.append(this_item)

        # save results
        output_dir = f"{args.root_path}/results/{args.decomp_data}/decomp/ner"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/decompose_{args.ent_algo}.json"
        write_list(output_path, outputs)
from utils.score_utils import bleu_multiple
from utils.utils import read_data, get_eval_model
from utils.globals import *
from utils.param import str2bool

from sentence_transformers import SentenceTransformer, util
import os
from tqdm import tqdm
import random
import argparse

parent_dir = os.path.dirname(os.path.abspath(__file__))
_ROOT_PATH = os.path.dirname(os.path.dirname(parent_dir))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_model", type=str, default="google/flan-t5-large")
    parser.add_argument("--flu_model", type=str, default="gpt-4o")
    parser.add_argument("--attr_model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=1,
        help = "temperature for text generation")
    parser.add_argument("--max_tokens", type=int, default=1000,
        help = "max new token for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=500,
        help = "max new token for text generation")
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()
    model_name = args.gen_model.split("/")[-1]
    data_dir = f"{_ROOT_PATH}/results/p2f/replace/{model_name}"
    files = os.listdir(data_dir)
    flu_mod = get_eval_model(args, args.flu_model)
    attr_mod = get_eval_model(args, args.attr_model)
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')
    for file_name in files:
        # if "epoch_1" not in file_name:
        #     continue
        file_path = f"{data_dir}/{file_name}"
        dataset = read_data(file_path)
        random.shuffle(dataset)
        dataset = dataset[:50]
        flu_scores = []
        sim_scores = []
        with tqdm(total=len(dataset)) as pbar:
            for item in dataset:
                gen_rpl = item["gen replacement"]
                query = item["raw query"]
                # compute the fluency score
                prompt = fluency_single_template.format(sentence=gen_rpl)
                # print(prompt)
                # print(prompt)
                success = False
                n_tries = 0
                while not success:
                    score = flu_mod.generate([prompt])[0]
                    # print(score)
                    try:
                        score = int(score)
                        if score in [1,2,3,4]:
                            success = True
                        else:
                            n_tries += 1
                    except Exception as e:
                        n_tries += 1
                    
                    if success or n_tries >= 5:
                        break
                    
                if not success:
                    pbar.update(1)
                    continue
                
                flu_scores.append(score)

                # compute the similarity score
                # extract corresponding attributes
                attributes = item["attributes"]
                max_sim = 0
                for attr in attributes:
                    prompt = extract_phrase_template.format(raw_query=query, attribute=attr, replace_query=gen_rpl)
                    rpl_attr = attr_mod.generate([prompt])[0]
                    # print(prompt)
                    # print(rpl_attr)
                    emb1 = sim_model.encode(attr)
                    emb2 = sim_model.encode(rpl_attr)
                    cos_sim = util.cos_sim(emb1, emb2)
                    cos_sim = cos_sim[0][0].item()
                    max_sim = max(max_sim, cos_sim)
                
                sim_scores.append(max_sim)
                pbar.update(1)
                avg_flu = sum(flu_scores)/len(flu_scores)
                avg_sim = sum(sim_scores)/len(sim_scores)
                pbar.set_postfix(flu_score=avg_flu, sim_score=avg_sim)

        print(f"Result for {file_name}")
        print(f"Average fluency score: {avg_flu}, average simliarity score: {avg_sim}")
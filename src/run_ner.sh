#!/bin/bash
# ent_algos=("spacy" "flair" "bert")
ent_algos=("bert")
for algo in ${ent_algos[@]}
do
    echo "Conduct ner extract using algorithm $algo"
    python ner.py \
        --decomp_data strategyQA --ent_algo=$algo
done
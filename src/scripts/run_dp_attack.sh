#!/bin/bash

dp_types=("text2text" "paraphrase")
eps_list=(0.1 0.01)

for dp in ${dp_types[@]}
do
    for eps in ${eps_list[@]}
    do
        echo "attribute inference attack: dp type $dp epsilon $eps"
        python -m attacks.dp_attack --dp_type=$dp --epsilon=$eps --attack_type attribute
    done
done

# for dp in ${dp_types[@]}
# do
#     for eps in ${eps_list[@]}
#     do
#         echo "reconstruction attack: dp type $dp epsilon $eps"
#         python -m attacks.dp_attack --dp_type=$dp --epsilon=$eps --attack_type reconstruct --gen_sample False
#     done
# done
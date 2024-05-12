#!/bin/bash

# dp_types=("text2text" "paraphrase")
# eps_list=(0.1 0.01)

# for dp in ${dp_types[@]}
# do
#     for eps in ${eps_list[@]}
#     do
#         echo "attribute inference attack: dp type $dp epsilon $eps"
#         python -m attacks.dp_attack --dp_type=$dp --epsilon=$eps --attack_type attribute
#     done
# done

# dp_types=("text2text" "paraphrase")

# for dp in ${dp_types[@]}
# do
#     echo "attribute inference attack: dp type $dp epsilon inf"
#     python -m attacks.dp_attack --dp_type=$dp
# done

# echo "attribute inference attack without sampling"
# python -m attacks.dp_attack --sample_train False

echo "attribute inference attack with sampling"
python -m attacks.dp_attack --sample_train True --epochs 10
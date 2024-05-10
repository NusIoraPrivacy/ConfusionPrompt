#!/bin/bash
# echo "Train decompose model"
# python train_decomp.py

# epos=(1 2 3 4 5)
# for epo in ${epos[@]}
# do
#     echo "Test decompose model for epoch $epo"
#     python test_decomp.py \
#         --test_epoch=$epo
# done
# datasets=("strategyQA" "musique")
datasets=("musique")
for data in ${datasets[@]}
do
    echo "Test decompose model for datasets $data"
    python test_decomp.py \
        --decomp_data=$data
done
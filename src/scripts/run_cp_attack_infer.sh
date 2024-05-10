#!/bin/bash

mus=(0.2 0.1 0.066 0.05)
sim_thds=(0.5 0.6 0.7 0.8)
flu_thds=(2 3)

for mu in ${mus[@]}
do
    for sim_thd in ${sim_thds[@]}
    do
        for flu_thd in ${flu_thds[@]}
        do
            echo "mu $mu sim_thd $sim_thd flu_thd $flu_thd"
            python -m attacks.cp_attack --mu=$mu --sim_thd=$sim_thd --flu_thd=$flu_thd --attack_type attribute
        done
    done
done
#!/bin/bash

mus=(0.066 0.05)

for mu in ${mus[@]}
do
    echo "tweet inference attack for mu $mu"
    python -m attacks.cp_tweet_infer --mu=$mu
done
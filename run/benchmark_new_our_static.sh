#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################

ds+=("walmart_norm")
ds+=("temperature_norm")

seeds=( 0 1 2 )

for seed in "${seeds[@]}"
do
    for d in "${ds[@]}"
    do
        cmd="python -m experiments.benchmark --ds $d --model NAR --run val_test --epochs 400 --seed $seed"
        echo $cmd
        eval $cmd
    done
done

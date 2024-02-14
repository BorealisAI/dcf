#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################

ds+=("walmart_norm")
ds+=("temperature_norm")

ms+=("DeepVAR")
ms+=("GPVAR")
ms+=("TransformerMAF")
ms+=("LSTMMAF")
ms+=("TimeGrad")

seeds=( 0 1 2 )

for seed in "${seeds[@]}"
do
    for d in "${ds[@]}"
    do
        for m in "${ms[@]}"
        do
            cmd="python -m experiments.benchmark --ds $d --model $m --run val_test --epochs 400 --seed $seed"
            echo $cmd
            eval $cmd
        done
    done
done

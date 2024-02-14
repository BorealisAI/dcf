#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################

ds+=("switch_ar_1")
ds+=("sin_ar_1")
ds+=("dynamic_ar_1")

ms+=("GroundTruth")
ms+=("DeepAR")
ms+=("DeepAR_10")
ms+=("DeepAR_160")
ms+=("DeepState")

seeds=( 0 1 2 )

run="val_test"

for seed in "${seeds[@]}"
do
    for d in "${ds[@]}"
    do
        for m in "${ms[@]}"
        do
            cmd="python -m experiments.synthetic --ds $d --model $m --run $run --seed $seed"
            echo $cmd
            eval $cmd
        done
    done
done

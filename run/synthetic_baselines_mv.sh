#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################

ds+=("dynamic_var_1")

ms+=("GroundTruth")
ms+=("DeepVAR")
ms+=("TransformerMAF")
ms+=("DeepVAR_10")
ms+=("DeepVAR_160")
ms+=("TransformerMAF_8")
ms+=("TransformerMAF_128")

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

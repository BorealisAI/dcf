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
ds+=("dynamic_var_1")

ms1+=("NAR")
ms1+=("MLPNAR")
ms2+=("NNAR")
ms2+=("MLPNNAR")

seeds=( 0 1 2 )

for seed in "${seeds[@]}"
do
    for d in "${ds[@]}"
    do
        for m in "${ms1[@]}"
        do
            cmd="python -m experiments.synthetic --ds $d --model $m --run val_test --seed $seed"
            echo $cmd
            eval $cmd
        done
        for m in "${ms2[@]}"
        do
            cmd="python -m experiments.synthetic --ds $d --model $m --run val_test --seed $seed"
            echo $cmd
            eval $cmd
        done
    done
done

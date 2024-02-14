#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################

ds+=("exchange_rate_nips")
ds+=("electricity_nips")
ds+=("solar_nips")
ds+=("traffic_nips")
ds+=("taxi_30min")
ds+=("wiki-rolling_nips")

seeds=( 0 1 2 )

for seed in "${seeds[@]}"
do
    for d in "${ds[@]}"
    do
        cmd="python -m experiments.benchmark --ds $d --model NNAR --run trainonval --baserun output/benchmark/NAR/trainonval --seed $seed"
        echo $cmd
        eval $cmd
    done
done

#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
python -m experiments.table.result \
    --split "test" \
    --ds "exchange_rate_nips" "solar_nips" "electricity_nips" "traffic_nips" "taxi_30min" "wiki-rolling_nips" \
    --modelrun NAR:trainonval NNAR:trainonval

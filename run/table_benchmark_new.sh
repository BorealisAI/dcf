#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
python -m experiments.table.result \
    --split "test" \
    --ds "walmart_norm" "temperature_norm" \
    --run val_test \
    --model DeepVAR GPVAR LSTMMAF TransformerMAF TimeGrad NAR NNAR

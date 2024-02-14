#!/bin/bash
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
python -m experiments.table.result --prefix synthetic \
    --split test \
    --ds switch_ar_1 sin_ar_1 dynamic_ar_1 dynamic_var_1 \
    --modelrun \
    GroundTruth:val_test \
    DeepAR:val_test DeepAR_10:val_test DeepAR_160:val_test \
    DeepState:val_test \
    DeepVAR:val_test DeepVAR_10:val_test DeepVAR_160:val_test \
    NAR:val_test MLPNAR:val_test \
    NNAR:val_test MLPNNAR:val_test

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
def freq_to_context_length(freq: str, prediction_length):
    if "H" in freq:
        context_length = 3 * prediction_length
    elif "D" in freq:
        context_length = 3 * prediction_length
    elif "T" in freq or "min" in freq:
        context_length = 3 * prediction_length
    else:
        context_length = prediction_length
    return context_length


def get_input_size(ds_name):
    if ds_name == "exchange_rate_nips":
        input_size = 28
    elif ds_name == "electricity_nips":
        input_size = 1484
    elif ds_name == "solar_nips":
        input_size = 552
    elif ds_name == "taxi_30min":
        input_size = 7290
    elif ds_name == "traffic_nips":
        input_size = 3856
    elif ds_name == "wiki-rolling_nips":
        input_size = 8002
    elif ds_name.startswith("walmart"):
        input_size = 94
    elif ds_name.startswith("temperature"):
        input_size = 3002
    else:
        raise NotImplementedError
    return input_size


def get_dim_time_feat(freq):
    if "H" in freq or "B" in freq or "W" in freq:
        return 4
    elif "D" in freq or "M" in freq:
        return 2
    else:
        return 6

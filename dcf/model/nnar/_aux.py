# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
# Copyright (c) 2020 Zalando SE
# This source code is licensed under the license found in the
# licenses/LICENSE_PyTorchTS file in the root directory of this source tree.
##########################################################################################
# code is based on the method from PyTorchTS: TransformerTempFlowTrainingNetwork
# https://github.com/zalandoresearch/pytorch-ts/blob/81be06bcc128729ad8901fcf1c722834f176ac34/pts/model/transformer_tempflow/transformer_tempflow_network.py#L97
##########################################################################################
import torch


def get_lagged_subsequences(
    sequence,
    sequence_length,
    indices,
    subsequences_length=1,
):
    assert max(indices) + subsequences_length <= sequence_length, (
        f"lags cannot go further than history length, found lag "
        f"{max(indices)} while history length is only {sequence_length}"
    )
    assert all(lag_index >= 0 for lag_index in indices)

    lagged_values = []
    for lag_index in indices:
        begin_index = -lag_index - subsequences_length
        end_index = -lag_index if lag_index > 0 else None
        lagged_values.append(sequence[:, begin_index:end_index, ...].unsqueeze(1))
    return torch.cat(lagged_values, dim=1).permute(0, 2, 3, 1)

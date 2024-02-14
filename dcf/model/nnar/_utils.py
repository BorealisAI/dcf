# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import torch


def create_time_ebmedding(D, T):
    assert D % 2 == 0
    wave_lens = torch.arange(0, D, 2) / D
    freq = torch.exp(torch.log(torch.tensor(T*2)) * (-wave_lens))
    pos = torch.arange(1, T+1)[:, None]
    t_embed = torch.cat((torch.sin(pos * freq), torch.cos(pos * freq)), dim=-1)
    return t_embed


def roll_sequence(y, l):
    y_roll = y.unfold(1, l, 1)
    return y_roll


def scale_transform(log_scale, eps=1e-8):
    return torch.exp(log_scale) + eps


def inverse_scale_transform(scale, eps=1e-8):
    return torch.log(torch.tensor(scale - eps))

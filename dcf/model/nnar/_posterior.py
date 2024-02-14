# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from ._utils import (
    scale_transform,
    inverse_scale_transform,
)


class PosteriorModel(nn.Module):
    def __init__(self, dim_target, dim_param, n_sample, **kwargs):
        super().__init__()
        self.dim_target = dim_target
        self.dim_param = dim_param
        self.n_sample = n_sample

    def get_dists_params(self, y, age, cond_sample, dim_idx):
        raise NotImplementedError

    def forward(self, y, age, compute_log_prob, cond_sample, dim_idx):
        dists, params = self.get_dists_params(y, age, cond_sample, dim_idx)
        param = torch.cat(params, dim=1)
        if compute_log_prob:
            log_prob = self.compute_log_prob(dists, params)
            return param, dists, log_prob
        else:
            return param, dists

    def compute_log_prob(self, dists, params):
        log_probs = []
        assert len(dists) == len(params)
        for t in range(0, len(dists)):
            log_probs.append(dists[t].log_prob(params[t]))
        log_prob = torch.cat(log_probs, dim=1)
        return log_prob


class ARMAPosteriorModel(PosteriorModel):
    def __init__(self, len_param, len_total, len_window=100, init_a=0.0, init_s=1e-3, use_nn=False, **kwargs):
        super().__init__(**kwargs)
        self.length = len_param
        self.len_total = len_total
        self.use_nn = use_nn
        if use_nn:
            self.dim_hidden = 512
            self.net = nn.Sequential(
                nn.Linear(self.dim_target, self.dim_hidden),
                nn.Tanh(),
                nn.Linear(self.dim_hidden, self.dim_hidden),
                nn.Tanh(),
                nn.Linear(self.dim_hidden, self.dim_target * self.dim_param)
            )
        else:
            self.m = nn.Parameter(torch.zeros((self.length, self.dim_target, self.dim_param)))
        self.s = nn.Parameter(torch.full((self.length, self.dim_target, self.dim_param), inverse_scale_transform(init_s)))
        # all params in the same target dim share the same a
        self.a = nn.Parameter(torch.full((self.length-1, self.dim_target, 1), float(init_a)))
        self.len_window = len_window


    def forward(self, y, age, compute_log_prob, cond_sample, dim_idx=None):
        N = age.shape[0]
        T = self.length
        if self.use_nn:
            m = self.net(y.squeeze(0))
            m = m.view(*m.shape[:-1], self.dim_target, self.dim_param)
        else:
            m = self.m
        s = self.s
        a = self.a
        m = m[:, dim_idx]
        s = scale_transform(s[:, dim_idx])
        a = torch.sigmoid(a[:, dim_idx])
        K = self.len_window
        param, log_prob = self.first_window(m[:K], s[:K], a[:K-1], N)
        params = [param]
        log_probs = [log_prob]
        for l in range(K, T, K):
            r = min(l + K, T)
            param, log_prob = self.next_window(m[l:r], s[l:r], a[(l-1):(r-1)], param[:, -1])
            params.append(param)
            log_probs.append(log_prob)
        param = torch.cat(params, 1)
        log_prob = torch.cat(log_probs, 1)
        if compute_log_prob:
            return param, None, log_prob
        else:
            return param, None

    def first_window(self, m, s, a, N):
        K = self.len_window
        a_pad = F.pad(a, (0, 0, 0, 0, K-1, 0))
        dist = Normal((1-a_pad)[K-2:] * m, s)
        z = dist.rsample(torch.Size([N * self.n_sample]))
        log_prob = dist.log_prob(z)
        b = a_pad.unfold(0, K-1, 1)
        c = b.flip(-1).cumprod(dim=-1).flip(-1)
        d0 = torch.ones_like(c[..., :1])
        d = torch.cat((c, d0), dim=-1)
        z = F.pad(z, (0, 0, 0, 0, K-1, 0))
        z = z.unfold(1, K, 1)
        param = z[..., None, :] @ d[..., None]
        param = param.view(log_prob.shape)
        return param, log_prob

    def next_window(self, m, s, a, cond_sample):
        S = cond_sample.shape[0]
        K = self.len_window
        dist = Normal((1-a) * m, s)
        z = dist.rsample([S])
        log_prob = dist.log_prob(z)
        # a[0] is only used for cond_sample
        a_pad = F.pad(a[1:], (0, 0, 0, 0, K-1, 0))
        b = a_pad.unfold(0, K-1, 1)
        c = b.flip(-1).cumprod(dim=-1).flip(-1)
        d0 = torch.ones_like(c[..., :1])
        d = torch.cat((c, d0), dim=-1)
        z = F.pad(z, (0, 0, 0, 0, K-1, 0))
        z = z.unfold(1, K, 1)
        param = z[..., None, :] @ d[..., None]
        param = param.view(log_prob.shape)
        a_prod = a.cumprod(0)
        # add a_prod * cond_sample at all t for all samples
        param += a_prod[None, :] * cond_sample[:, None]
        return param, log_prob

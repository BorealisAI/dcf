# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import torch
import torch.nn as nn
from torch.distributions import Normal, MixtureSameFamily, Categorical, Independent


class PriorModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, param_sample, compute_log_prob=False, cond_sample=None, dim_idx=None):
        log_p = []
        dists = []
        if cond_sample is None:
            dist_0 = self.get_init_dist(param_sample.shape[0], dim_idx=dim_idx)
            dists.append(dist_0)
            if compute_log_prob:
                log_p.append(dist_0.log_prob(param_sample[:, 0:1]))
            if param_sample.shape[1] > 1:
                dist_t = self.get_next_dist(param_sample[:, :-1, :], dim_idx=dim_idx)
                dists.append(dist_t)
            t0 = 1
        else:
            dist_t = self.get_next_dist(torch.cat((cond_sample, param_sample[:, :-1]), dim=1), dim_idx=dim_idx)
            dists.append(dist_t)
            t0 = 0
        if compute_log_prob:
            if param_sample.shape[1] > t0:
                log_p.append(dist_t.log_prob(param_sample[:, t0:]))
            log_p = torch.cat(log_p, dim=-2)
            return dists, log_p
        else:
            return dists 
    
    def sample(self, prev_param_sample, len_pred, dim_idx):
        param_samples = []
        for t in range(len_pred):
            dist = self.get_next_dist(prev_param_sample, dim_idx=dim_idx)
            sample_t = dist.sample()
            param_samples.append(sample_t)
            prev_param_sample = sample_t
        param_sample = torch.cat(param_samples, dim=1)
        return param_sample

    def get_init_dist(self, batch_size):
        raise NotImplementedError
    
    def get_next_dist(self, param_sample):
        raise NotImplementedError


class SimpleMixPriorModel(PriorModel):
    def __init__(
        self,
        dim_target,
        dim_param,
        sigma_ub=1.0,
        sigma_d_ub=1e-1,
        lambd_ub=1.0,
        lambd_lb=0.0,
    ):
        super().__init__()
        self.dim_target = dim_target
        self.dim_param = dim_param
        log_sigma_0 = torch.zeros((dim_target, dim_param))
        log_sigma_d = torch.zeros((dim_target, dim_param))
        log_lambd = torch.zeros((dim_target,))
        self.register_buffer("mu_0", torch.zeros((1, 1)))
        self.register_buffer("sigma_ub", torch.tensor(sigma_ub))
        self.log_sigma_0 = nn.Parameter(log_sigma_0)
        self.log_sigma_d = nn.Parameter(log_sigma_d)
        self.log_lambd = nn.Parameter(log_lambd)
        self.register_buffer("lambd_ub", torch.tensor(lambd_ub))
        self.register_buffer("lambd_lb", torch.tensor(lambd_lb))
        self.register_buffer("sigma_d_ub", torch.tensor(sigma_d_ub))
        self.dist = Normal
    
    def get_scale_d(self, dim_idx):
        return self.sigma_d_ub * torch.sigmoid(self.log_sigma_d[dim_idx, :])

    def get_scale_0(self, dim_idx):
        return self.sigma_ub * torch.sigmoid(self.log_sigma_0[dim_idx, :])

    def get_lambd(self, dim_idx):
        return self.lambd_lb + (self.lambd_ub - self.lambd_lb) * torch.sigmoid(self.log_lambd[dim_idx])
    
    def get_init_dist(self, batch_size, dim_idx):
        dist_0 = self.dist(
            self.mu_0[None, None, :, :].expand(batch_size, 1, dim_idx.shape[-1], self.dim_param),
            self.get_scale_0(dim_idx)
        )
        return dist_0
    
    def get_next_dist(self, param_sample, dim_idx):
        N = param_sample.shape[0]
        T = param_sample.shape[1]
        scale_d = self.get_scale_d(dim_idx)
        scale_0 = self.get_scale_0(dim_idx)
        scale = torch.stack((scale_0, scale_d), dim=-2)
        lambd = self.get_lambd(dim_idx)[None, None, :]
        mix_dist = Categorical(
            torch.stack((lambd, 1.0-lambd), dim=-1).expand(N, T, -1, -1)
        )
        mu = torch.stack(
            (
                self.mu_0[None, None, :, :].expand(N, T, dim_idx.shape[-1], self.dim_param),
                param_sample,
            ),
            dim=-2)
        comp_dist = Independent(self.dist(mu, scale), 1)
        return MixtureSameFamily(mix_dist, comp_dist)

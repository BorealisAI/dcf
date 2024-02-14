# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import math

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal, Categorical, MultivariateNormal 
from pts.modules.flows import FlowSequential

from ._utils import (
    scale_transform,
    inverse_scale_transform,
    roll_sequence,
    create_time_ebmedding,
)
from ._aux import (
    get_lagged_subsequences
)
from ._prior import *
from ._posterior import *


class ShiftScaleLayer(nn.Module):
    def __init__(self, m, s):
        super().__init__()
        m = torch.tensor(m)
        s = torch.tensor(s)
        if (s > 0).any():
            i = (s == 0).nonzero()
            print("[WARNING] following dimensions have zero std:")
            print(i)
            s[i] = 1.0
        self.register_buffer("m", m)
        self.register_buffer("s", s)
        self.register_buffer("log_s", torch.log(s))
    
    def forward(self, y, *args, dim_idx=None):
        if dim_idx is None:
            return (y - self.m) / self.s, -self.log_s[None, None, :].expand(*y.shape)
        else:
            return (y - self.m[dim_idx]) / self.s[dim_idx], -self.log_s[None, None, dim_idx].expand(*y.shape)
    
    def inverse(self, z, *args, dim_idx=None):
        if dim_idx is None:
            return z * self.s + self.m, self.log_s[None, None, :].expand(*z.shape)
        else:
            return z * self.s[dim_idx] + self.m[dim_idx], self.log_s[None, None, dim_idx].expand(*z.shape)


def unsqueeze_as(a, b):
    n_dim_diff = len(b.shape) - len(a.shape)
    if n_dim_diff > 0:
        return a[(...,) + (None,)*n_dim_diff]
    else:
        assert n_dim_diff == 0
        return a


class IdentityTransform:
    def __call__(self, y, dim_idx=None):
        return y, torch.zeros_like(y)

    def inverse(self, y, dim_idx=None):
        return y, torch.zeros_like(y)


class TransformSequential(nn.Sequential):
    def set_param(self, y, time_idx=None):
        for module in self:
            module.set_param(y, time_idx)
            y, _ = module(y, time_idx)

    def forward(self, y, dim_idx=None):
        sum_log_det = 0
        for module in self:
            x, log_det = module(y, dim_idx)
            sum_log_det += log_det
        return x, sum_log_det

    def inverse(self, z, dim_idx=None):
        sum_log_det = 0
        for module in reversed(self):
            z, log_det = module.inverse(z, dim_idx)
            sum_log_det += log_det
        return z, sum_log_det


class ShiftScaleTransform(nn.Module):
    def __init__(self, global_std):
        super().__init__()
        self.m = None
        self.s = None
        self.register_buffer("global_std", torch.tensor(global_std).float())
    
    def set_param(self, y, time_idx=None, eps=1e-4):
        m = y.mean(-1)
        s = y.std(-1)
        if time_idx is not None:
            m = m[:, time_idx]
            s = s[:, time_idx]
        self.m = m
        self.s = torch.where(
            s >= eps,
            s,
            self.global_std,
        )
    
    def forward(self, y, dim_idx=None):
        if dim_idx is None:
            m = unsqueeze_as(self.m, y)
            s = unsqueeze_as(self.s, y)
        else:
            m = unsqueeze_as(self.m[..., dim_idx], y)
            s = unsqueeze_as(self.s[..., dim_idx], y)
        return (y - m) / s, -s.log()
    
    def inverse(self, z, dim_idx=None):
        if dim_idx is None:
            m = unsqueeze_as(self.m, z)
            s = unsqueeze_as(self.s, z)
        else:
            m = unsqueeze_as(self.m[..., dim_idx], z)
            s = unsqueeze_as(self.s[..., dim_idx], z)
        return z * s + m, s.log()


class QuantileTransform(nn.Module):
    def __init__(self, p=[0.02, 0.95]):
        super().__init__()
        print(f"Quantile transforme: {p}")
        self.register_buffer("p", torch.tensor(p).float())
    
    def set_param(self, y, time_idx=None):
        q0 = torch.quantile(y, self.p[0], dim=-1, keepdim=False, interpolation="higher")
        q1 = torch.quantile(y, self.p[1], dim=-1, keepdim=False, interpolation="lower")
        if time_idx is not None:
            # q has extra dim 0
            q0 = q0[:, time_idx]
            q1 = q1[:, time_idx]
        self.q = (q0, q1)
    
    def forward(self, y, dim_idx=None):
        q0, q1 = self.q
        if dim_idx is not None:
            q0 = q0[..., dim_idx]
            q1 = q1[..., dim_idx]
        q0 = unsqueeze_as(q0, y)
        q1 = unsqueeze_as(q1, y)
        y = torch.where(
            y < q0,
            q0,
            y
        )
        y = torch.where(
            y > q1,
            q1,
            y,
        )
        return y, torch.zeros_like(y)
    
    def inverse(self, z, dim_idx=None):
        return z, torch.zeros_like(z)


class LocScaleDist(nn.Module):
    def __init__(self, dim, scale=None, constant_scale=False, dist=Normal):
        super().__init__()
        if scale is None:
            if constant_scale:
                s = torch.tensor([inverse_scale_transform(1.0)] * dim)
                self.s = nn.Parameter(s)
                self.fix_s = True
                self.dim_input = dim
            else:
                self.fix_s = False
                self.dim_input = dim * 2
        else:
            self.register_buffer("s", torch.tensor([inverse_scale_transform(scale)]))
            self.fix_s = True
            self.dim_input = dim
        self.scale = None
        self.dist = dist
    
    def get_sigma(self, dim_idx, cond_scale=None):
        if self.fix_s:
            return scale_transform(self.s[dim_idx])
        else:
            return scale_transform(cond_scale)

    def forward(self, cond, dim_idx, cond_scale=None):
        if self.fix_s:
            m = cond
            s = scale_transform(self.s[dim_idx])
            return self.dist(m, s)
        else:
            m = cond
            s = scale_transform(cond_scale)
            return self.dist(m, s)

    def log_prob(self, x, cond, dim_idx, cond_scale=None):
        log_p = 0.0
        if self.scale is not None:
            scale = self.scale[..., dim_idx]
            x = x / scale
            log_p = log_p - torch.log(scale)
        log_p = log_p + self(cond, dim_idx, cond_scale).log_prob(x)
        return log_p
    
    def sample(self, dim_idx, sample_shape=torch.Size(), cond=None, cond_scale=None):
        x = self(cond, dim_idx, cond_scale).sample(sample_shape)
        if self.scale is not None:
            x = x * self.scale[..., dim_idx]
        return x


class IdentityEncoder(nn.Module):
    def forward(self, x, x_p, *args):
        # x: (NT, L, C)
        return x_p.reshape(x_p.shape[0], -1)


class SimpleEncoder(nn.Module):
    def __init__(self, dim_input):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim_input))
        self.shift = nn.Parameter(torch.zeros(dim_input))
    
    def forward(self, x, x_p, *args):
        return torch.tanh(self.scale * x_p + self.shift)


class MLPEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dim_output):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=dim_input,
                out_features=dim_hidden,
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=dim_hidden,
                out_features=dim_output,
            ),
            nn.Tanh(),
        )

    def forward(self, x, x_p, *args):
        return self.encoder(x_p)


class LSTMEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, dropout_rate, **kwargs):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=dim_input,
            hidden_size=dim_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate,
        )
    
    def forward(self, x_ctx, x_pred, *args):
        h, _ = self.encoder(
            torch.cat((x_ctx, x_pred), dim=1)
        )
        return h[:, -1]


class TransformerEncoder(nn.Module):
    def __init__(self, dim_input, dim_hidden, len_context, n_head=4, **kwargs):
        super().__init__()
        d_model = dim_hidden
        dim_t = d_model
        t_embed = create_time_ebmedding(dim_t, len_context+1)
        self.register_buffer("t_embed", t_embed)
        self.prelayer = nn.Linear(dim_input + dim_t, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=4*d_model,
            num_encoder_layers=3,
            num_decoder_layers=3,
            batch_first=True,
        )
    
    def forward(self, x_ctx, x_pred, *args):
        t_embed = self.t_embed[None, :, :].expand(x_ctx.shape[0], -1, -1)
        x_ctx = self.prelayer(torch.cat((x_ctx, t_embed[:, :-1]), dim=-1))
        x_pred = self.prelayer(torch.cat((x_pred, t_embed[:, -1:]), dim=-1))
        out = self.transformer(x_ctx, x_pred)
        return out
    

class DynamicToCond(nn.Module):
    def __init__(self, dim_input, dim_cond):
        super().__init__()
        self.w = nn.Parameter(torch.rand(dim_cond, dim_input))
        self.b = nn.Parameter(torch.zeros(dim_cond)) 
        self.dim_input = dim_input
        self.dim_cond = dim_cond

    def forward(self, z, dim_idx):
        z = z.reshape(*z.shape[:-1], self.dim_cond, self.dim_input)
        y = (z[..., dim_idx, :] * self.w[dim_idx, :]).sum(-1) + self.b[dim_idx]
        return y


class NARBaseNetwork(nn.Module):
    def __init__(
        self,
        dim_target,
        len_pred,
        len_context,
        len_pred_scale,
        len_context_scale,
        len_hist,
        len_infer_model,
        lags_seq,
        dim_input=None,
        dim_hidden=256,
        dim_time_feat=None,
        use_identity=False,
        dim_dynamic_in=None,
        scaling=None,
        invnet=None,
        invnet_out=None,
        dist="Normal",
        dequantize=None,
        encoder="LSTM",
        dropout_rate=0.0,
        constant_scale=False,
        opts=[],
        **kwargs,
    ):
        super().__init__()
        self.len_pred = len_pred
        self.len_context = len_context
        self.len_pred_scale = len_pred_scale
        self.len_context_scale = len_context_scale
        self.len_hist = len_hist
        self.len_infer_model = len_infer_model
        self.lags_seq = lags_seq
        self.dim_target = dim_target
        self.dim_time_feat = dim_time_feat
        print(f"dim_dynamic_in = {dim_dynamic_in}")
        self.dim_dynamic_in = dim_dynamic_in
        self.use_identity = use_identity
        print(f"use identity = {self.use_identity}")
        print(f"dist: {dist}")
        print(f"constant scale = {constant_scale}")
        self.dim_cond_scale = None
        if dist == "Normal":
            self.dist = LocScaleDist(
                dim_target,
                scale=None,
                constant_scale=constant_scale,
                dist=Normal,
            )
            self.dim_cond = dim_target
            if not constant_scale:
                self.dim_cond_scale = dim_target
        else:
            raise NotImplementedError
        print(f"encoder: {encoder}")
        print(f"dim_cond = {self.dim_cond}")
        print(f"dim_hidden = {dim_hidden}")
        if self.use_identity:
            dim_input = dim_target * len_context
            if encoder == "MLP":
                self.encoder = MLPEncoder(
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    dim_output=dim_hidden,
                )
                self.dim_embed = dim_hidden
            else:
                if "id" in opts:
                    self.encoder = IdentityEncoder()
                else:
                    self.encoder = SimpleEncoder(dim_input)
                self.dim_embed = dim_input
        else:
            assert dim_input is not None
            if encoder == "LSTM":
                self.encoder = LSTMEncoder(
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    dropout_rate=dropout_rate,
                )
            elif encoder == "Transformer":
                self.encoder = TransformerEncoder(
                    dim_input=dim_input,
                    dim_hidden=dim_hidden,
                    len_context=len_context,
                    **kwargs,
                )
            else:
                raise NotImplementedError
            self.dim_embed = dim_hidden
            self.dim_embed_index = 1
            self.embed_index = nn.Embedding(
                num_embeddings=self.dim_target,
                embedding_dim=self.dim_embed_index,
            )
        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        print(f"dim_cond_scale = {self.dim_cond_scale}")
        if self.dim_cond_scale:
            self.encoder_scale = nn.Sequential(
                nn.Linear(dim_time_feat, dim_hidden),
                nn.Tanh(),
                nn.Linear(dim_hidden, self.dim_dynamic_in * self.dim_target),
                nn.Tanh(),
            )
            self.dynamic_to_cond_scale = DynamicToCond(self.dim_dynamic_in, self.dim_cond_scale)
        self.dequantize = dequantize
        print(f"dequantize = {dequantize}")
        if not scaling:
            scaling = IdentityTransform()
        self.scaling = scaling
        print(f"scaling = {scaling}")
        print(f"use invnet = {invnet is not None}")
        if invnet is not None:
            print(invnet)
            if isinstance(invnet, list):
                self.invnet = FlowSequential(*invnet)
            else:
                self.invnet = invnet
        else:
            self.invnet = None
        if invnet_out is not None:
            print("invnet_out:")
            print(invnet_out)
            if isinstance(invnet_out, list):
                self.invnet_out = FlowSequential(*invnet_out)
            else:
                self.invnet_out = invnet_out
        else:
            self.invnet_out = None
    
    def cond_scale(self, x_p, dim_idx):
        if self.dim_cond_scale is None:
            return None
        else:
            return self.dynamic_to_cond_scale(self.encoder_scale(x_p[..., 0, -self.dim_time_feat:]), dim_idx)
    
    def dequantize_target(self, y, dim_idx):
        if self.dequantize is not None:
            noise = torch.empty_like(y).uniform_(*self.dequantize)
            if self.invnet is not None:
                cond = None
                y_org, _ = self.invnet.inverse(y, cond, dim_idx=dim_idx)
                y_noise = y_org + noise
                return self.invnet(y_noise, cond, dim_idx=dim_idx)[0]
            else:
                return y + noise
        else:
            return y
    
    def transform_target(
        self,
        target,
        cond,
    ):
        cond = None
        if self.invnet is not None:
            target, log_det_jac = self.invnet(target, cond)
            return target, log_det_jac
        else:
            return target, None
    
    def inverse_transform_target(
        self,
        target,
        cond,
    ):
        cond = None
        if self.invnet is not None:
            target, log_det_jac = self.invnet.inverse(target, cond)
            return target, log_det_jac
        else:
            return target, None
    
    def transform_output(self, target, dim_idx, dequantize):
        if dequantize:
            target = self.dequantize_target(target, dim_idx)
        target, log_det_jac = self.scaling(target, dim_idx)
        if self.invnet_out is not None:
            target, log_det_jac_inv = self.invnet_out(target, cond=None)
            log_det_jac = log_det_jac + log_det_jac_inv
        return target, log_det_jac
    
    def inverse_transform_output(self, target, dim_idx):
        log_det_jac = 0
        if self.invnet_out is not None:
            target, log_det_jac_inv = self.invnet_out.inverse(target, cond=None)
            log_det_jac = log_det_jac + log_det_jac_inv
        target, log_det_jac_scale = self.scaling.inverse(target, dim_idx)
        log_det_jac = log_det_jac + log_det_jac_scale
        return target, log_det_jac
    
    def transform_loglike(
        self,
        LL,
        log_det_jac_full,
        log_det_jac_out,
        dim_idx,
        len_pred,
        sample_size=None,
        time_idx=None,
    ):
        LL = LL.sum(-1)
        if log_det_jac_full is not None:
            log_det_jac_full = log_det_jac_full[:, -len_pred:, dim_idx]
            if time_idx is not None:
                log_det_jac_full = log_det_jac_full[:, time_idx]
        for log_det_jac in [log_det_jac_full, log_det_jac_out]:
            if log_det_jac is not None:
                log_det_jac = log_det_jac.sum(-1)
                if sample_size is not None:
                    log_det_jac = log_det_jac[:, None, :].expand(-1, sample_size, -1)
                LL = LL + log_det_jac
        return LL

    def set_local_transform(
        self,
        past_target,
        past_observed_values,
        past_is_pad,
        len_pred,
    ):
        if not isinstance(self.scaling, IdentityTransform):
            past_observed_values = torch.min(
                past_observed_values, 1 - past_is_pad.unsqueeze(-1)
            )
            assert past_observed_values.shape == past_target.shape
            assert (past_observed_values == 1).all()
            T = past_target.shape[1]
            L = self.len_context_scale
            P = self.len_pred_scale
            # skip last P steps to avoid using predictions to compute scale
            # note that len_pred (arg) may not be the same as P at test time
            r = T - P
            l = r - L - len_pred + 1
            rolled_past_target = roll_sequence(past_target[:, l:r], L)
            self.scaling.set_param(rolled_past_target)

    def create_input(
        self,
        past_target,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        len_pred,
        cache=None,
    ):
        L = self.len_context
        subsequences_length = len_pred + L

        if cache is not None:
            lags = cache.get("lags")
        else:
            lags = None
        if lags is None:
            # (batch_size, sub_seq_len, target_dim, num_lags)
            lags = get_lagged_subsequences(
                sequence=past_target,
                sequence_length=past_target.shape[1],
                indices=self.lags_seq,
                subsequences_length=len_pred + L,
            )
            # (batch_size, len_pred, target_dim, num_lags, L+1)
            lags = roll_sequence(lags, L + 1)
            if cache is not None:
                cache["lags"] = lags

        self.set_local_transform(
            past_target=past_target,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            len_pred=len_pred,
        )

        lags_scaled = self.scaling(lags)[0]

        input_lags = lags_scaled.reshape(
            (-1, len_pred, len(self.lags_seq) * self.dim_target, L + 1)
        )

        if not self.use_identity:
            if cache is not None:
                time_feat = cache.get("time_feat")
                rolled_index_embeddings = cache.get("index_embed")
            else:
                time_feat = None
                rolled_index_embeddings = None
            if time_feat is None or rolled_index_embeddings is None:
                time_feat = roll_sequence(past_time_feat[:, -subsequences_length:], L + 1)

                # (batch_size, target_dim, embed_dim)
                index_embeddings = self.embed_index(target_dimension_indicator)

                # (batch_size, seq_len, target_dim * embed_dim)
                repeated_index_embeddings = (
                    index_embeddings.unsqueeze(1)
                    .expand(-1, subsequences_length, -1, -1)
                    .reshape((-1, subsequences_length, self.dim_target * self.dim_embed_index))
                )

                rolled_index_embeddings = roll_sequence(repeated_index_embeddings, L+1)
                if cache is not None:
                    cache["time_feat"] = time_feat
                    cache["index_embed"] = rolled_index_embeddings

            # (batch_size, sub_seq_len, input_dim)
            inputs = torch.cat((input_lags, rolled_index_embeddings, time_feat), dim=-2)
        else:
            inputs = input_lags

        # (N, T, C, L) -> (N, T, L, C)
        return inputs.permute(0, 1, 3, 2)

    def forward(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
    ):
        if self.training:
            return self.training_forward(
                past_target,
                past_feat_dynamic_age,
                past_time_feat,
                past_observed_values,
                past_is_pad,
                target_dimension_indicator,
            )
        else:
            return self.eval_forward(
                past_target,
                past_feat_dynamic_age,
                past_time_feat,
                past_observed_values,
                past_is_pad,
                target_dimension_indicator,
            )


class NARTrainingNetwork(NARBaseNetwork):
    def __init__(
        self,
        n_sample,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_sample = n_sample
        dim_dynamic_in = self.dim_dynamic_in
        if dim_dynamic_in is not None:
            self.embed_to_dynamic = nn.Sequential(
                nn.Linear(self.dim_embed, dim_dynamic_in * self.dim_cond),
                nn.Tanh()
            )
            self.dynamic_to_cond = DynamicToCond(dim_dynamic_in, self.dim_cond)
            self.embed_to_cond = lambda x, dim_idx: self.dynamic_to_cond(self.embed_to_dynamic(x), dim_idx)
        else:
            self.embed_to_cond = nn.Linear(self.dim_embed, self.dim_cond)

    def get_encoding(self, x, x_p, dim_idx):
        # shape: N, T, L, C
        N = x.shape[0]
        T = x.shape[1]
        assert x.shape[2] == self.len_context
        h = self.encoder(
            x.reshape(N * x.shape[1], x.shape[2], x.shape[3]),
            x_p.reshape(N * x_p.shape[1], x_p.shape[2], x_p.shape[3]),
        )
        h = h.reshape(N, T, h.shape[-1])
        return self.embed_to_cond(h, dim_idx)

    def training_forward(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
    ):
        dim_idx = torch.arange(self.dim_target, device=past_target.device)
        len_pred = self.len_infer_model
        past_target, log_det_jac = self.transform_target(past_target, past_time_feat)
        # create input and output
        x_roll = self.create_input(
            past_target=past_target,
            past_time_feat=past_time_feat,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            target_dimension_indicator=target_dimension_indicator,
            len_pred=len_pred,
        )
        x_in = x_roll[:, :, :-1]
        x_p = x_roll[:, :, -1:]
        cond = self.get_encoding(x_in, x_p, dim_idx)
        y_out = past_target[:, -len_pred:, dim_idx]
        y_out, log_det_jac_out = self.transform_output(y_out, dim_idx, dequantize=self.dequantize)
        LL = self.dist.log_prob(
            y_out, cond, dim_idx,
            cond_scale=self.cond_scale(x_p, dim_idx),
        )
        LL = self.transform_loglike(LL, log_det_jac, log_det_jac_out, dim_idx, len_pred)
        return (-LL.mean(0).sum(0),)

    def eval_forward(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
    ):
        return self.training_forward(
            past_target,
            past_feat_dynamic_age,
            past_time_feat,
            past_observed_values,
            past_is_pad,
            target_dimension_indicator,
        )
    
    def predict(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        future_time_feat,
        future_target=None,
    ):
        dim_idx = torch.arange(self.dim_target, device=past_target.device)
        past_target, _ = self.transform_target(past_target, past_time_feat)
        L = self.len_context
        H = self.len_hist
        past_target = past_target[:, -H:]
        past_time_feat = past_time_feat[:, -H:]
        past_observed_values = past_observed_values[:, -H:]
        assert (past_observed_values == 1.0).all()
        past_is_pad = past_is_pad[:, -H:]
        assert (past_is_pad == 0.0).all()
        y = torch.repeat_interleave(past_target, self.n_sample, dim=0)
        time_feat = torch.repeat_interleave(past_time_feat, self.n_sample, dim=0)
        future_time_feat = torch.repeat_interleave(future_time_feat, self.n_sample, dim=0)
        target_dimension_indicator = torch.repeat_interleave(target_dimension_indicator, self.n_sample, dim=0)
        y_out = []
        y_pad = torch.zeros_like(y[:, -1:])
        if future_target is not None:
            future_target, log_det_jac = self.transform_target(future_target, None)
            future_target = torch.repeat_interleave(future_target, self.n_sample, dim=0)
            loglike = []
        for t in range(self.len_pred):
            time_feat = torch.cat((time_feat[:, 1:], future_time_feat[:, t:t+1]), dim=1)
            past_target_t = torch.cat((y, y_pad), dim=1)
            x_roll = self.create_input(
                past_target=past_target_t,
                past_time_feat=time_feat,
                past_observed_values=torch.ones_like(past_target_t),
                past_is_pad=torch.zeros(*past_target_t.shape[:-1], device=past_target_t.device),
                target_dimension_indicator=target_dimension_indicator,
                len_pred=1,
            )
            x_in = x_roll[:, :, :-1]
            x_p = x_roll[:, :, -1:]
            cond = self.get_encoding(x_in, x_p, dim_idx=dim_idx)
            y_out_t = self.dist.sample(cond=cond, dim_idx=dim_idx, cond_scale=self.cond_scale(x_p, dim_idx))
            y_out_t, _ = self.inverse_transform_output(y_out_t, dim_idx)
            y_out.append(y_out_t)
            if future_target is not None:
                future_target_t = future_target[:, t:t+1]
                future_target_t, log_det_jac_out = self.transform_output(future_target_t, dim_idx, dequantize=False)
                LL = self.dist.log_prob(future_target_t, cond=cond, dim_idx=dim_idx, cond_scale=self.cond_scale(x_p, dim_idx))
                LL = self.transform_loglike(LL, log_det_jac, log_det_jac_out, dim_idx, len_pred=1)
                loglike.append(LL)
            # be careful when len_context == 1
            y = torch.cat((y[:, 1:], y_out[-1]), dim=1)
        y_out = torch.cat(y_out, dim=1)
        y_out, _ = self.inverse_transform_target(y_out, future_time_feat)
        if future_target is not None:
            return y_out, torch.cat(loglike, dim=1)
        else:
            return y_out


class NARPredictionNetwork(NARTrainingNetwork):
    def forward(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        future_time_feat,
    ):
        y_out = self.predict(
            past_target,
            past_feat_dynamic_age,
            past_time_feat,
            past_observed_values,
            past_is_pad,
            target_dimension_indicator,
            future_time_feat,
        )
        if self.dim_target == 1:
            # univariate prediction
            return y_out.reshape(
                -1,
                self.n_sample,
                self.len_pred,
            )
        else:
            return y_out.reshape(
                -1,
                self.n_sample,
                self.len_pred,
                self.dim_target,
            )


class NNARTrainingNetwork(NARBaseNetwork):
    def __init__(
        self,
        len_total,
        len_infer_post,
        n_sample_param,
        n_sample_dim=20,
        n_sample_time=None,
        dim_hidden_infer=2048,
        dim_param=None,
        step_param=1,
        penalty=None,
        kwargs_prior={},
        kwargs_posterior={},
        opts=[],
        **kwargs,
    ):
        super().__init__(opts=opts, **kwargs)
        self.len_infer_post = len_infer_post
        self.n_sample_param = n_sample_param
        self.n_sample_dim = n_sample_dim
        print(f"n_sample_dim = {n_sample_dim}")
        self.n_sample_time = n_sample_time
        assert self.dim_cond == self.dim_target
        if "id" in opts:
            self.embed_to_dynamic = lambda x: x
        else:
            self.embed_to_dynamic = nn.Sequential(
                nn.Linear(
                    self.dim_embed,
                    self.dim_dynamic_in * self.dim_target,
                ),
                nn.Tanh()
            )
        self.bias_cond = nn.Parameter(torch.zeros(self.dim_cond,))
        self.dynamic_to_cond = lambda x, dim_idx: x + self.bias_cond[dim_idx]
        dim_param = self.dim_dynamic_in
        self.param_shift = nn.Parameter(torch.rand(self.dim_target, dim_param))
        if "zero" in opts:
            self.param_scale = lambda x, dim_idx: torch.zeros(*x.shape[:2], len(dim_idx), 1, device=x.device)
        else:
            self.param_scale = lambda x, dim_idx: torch.ones(*x.shape[:2], len(dim_idx), 1, device=x.device) 
        def transform_param(param, x_p, dim_idx):
            scale = self.param_scale(x_p, dim_idx)
            param_scaled = param.view(x_p.shape[0], -1, *param.shape[1:]) * scale[:, None]
            return param_scaled.view(*param.shape) + self.param_shift[dim_idx, :]
        self.transform_param = transform_param
        self.penalty = None
        print(f"dim_param = {dim_param}")
        print(f"penalty = {penalty}")
        self.dim_param = dim_param
        self.prior = SimpleMixPriorModel(
            dim_target=self.dim_target,
            dim_param=self.dim_param,
            **kwargs_prior,
        )
        self.dim_hidden_infer = dim_hidden_infer
        self.len_total = len_total
        self.step_param = step_param
        self.len_param = math.ceil((self.len_total - self.len_hist) / self.step_param)
        print(f"step_param: {self.step_param}, len_param: {self.len_param}, len_total: {self.len_total}, len_hist: {self.len_hist}")
        posterior_cls = self.choose_posterior(opts)
        self.posterior = posterior_cls(
            dim_target=self.dim_target,
            dim_param=self.dim_param,
            dim_hidden=self.dim_hidden_infer,
            len_hist=self.len_hist,
            len_infer=self.len_infer_post,
            len_total=self.len_total,
            len_param=self.len_param,
            n_sample=self.n_sample_param,
            **kwargs_posterior,
        )
        self.fix_base_params(False)
    
    def load_base_state(self, base_state):
        model_state = self.state_dict()
        model_state.update({
            "param_shift": base_state["dynamic_to_cond.w"],
            "bias_cond": base_state["dynamic_to_cond.b"],
        })
        to_load = lambda name: (
            name.startswith("encoder.")
            or name.startswith("embed_index.")
            or name.startswith("embed_to_dynamic.")
            or name.startswith("encoder_scale.")
            or name.startswith("dynamic_to_cond_scale.")
            or self._is_dist_param(name)
        )
        base_state = {k:v for k,v in base_state.items() if k in model_state and to_load(k)}
        model_state.update(base_state)
        print("Load base model state:")
        print(list(base_state.keys()))
        self.load_state_dict(model_state)

    def fix_base_params(self, switch):
        self._fix_base = switch
        if switch:
            print("Base params fixed.")
            self._is_updatable_param = lambda name: not (
                name.startswith("encoder.")
                or name.startswith("embed_index.")
                or name.startswith("encoder_scale.")
                or name.startswith("dynamic_to_cond_scale.")
                or self._is_dist_param(name)
                or name.startswith("embed_to_dynamic.")
                or name.startswith("param_shift")
            )
        else:
            self._is_updatable_param = lambda _: True
        for n, p in self.named_parameters():
            if not self._is_updatable_param(n):
                p.requires_grad = False
    
    def choose_posterior(self, opts):
        posterior_cls = ARMAPosteriorModel
        return posterior_cls

    def select_samples_directly(self, samples, t):
        # samples: (S, len_total, C)
        # t: (N, T)
        N, T = t.shape
        S, _, O, I = samples.shape
        # index: (S, N*T, C)
        index = (t.long() - self.len_hist).reshape(-1)[None, :, None, None].expand(S, -1, O, I)
        selected = torch.gather(samples, 1, index).reshape(
                S, N, T, O, I
            ).permute(1, 0, 2, 3, 4).reshape(
                -1, T, O, I
            )
        # selected: (N*S, T, C)
        return selected

    def select_samples(self, samples, t):
        t = t.long()
        return self.select_samples_directly(samples, t)

    def sample_param_posterior_var(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat=None,
        past_observed_values=None,
        past_is_pad=None,
        target_dimension_indicator=None,
        cond_sample=None,
        start=0,
        last_only=False,
        dim_idx=None,
    ):
        if cond_sample is None:
            cond_sample = self.get_init_sample(past_feat_dynamic_age.shape[0], dim_idx)
        if dim_idx is None:
            dim_idx = torch.arange(self.dim_target, device=past_target.device)
        param_sample, dists_q, prob_q = self.posterior(
            past_target[:, start:],
            past_feat_dynamic_age[:, start:],
            True,
            cond_sample,
            dim_idx=dim_idx,
        )
        return param_sample, dists_q, prob_q
    
    def _is_dist_param(self, name):
        return name.startswith("dist")

    def _is_prior_param(self, name):
        return name.startswith("prior")
    
    def _is_post_param(self, name):
        return name.startswith("posterior")

    def _is_latent_param(self, name):
        return (
            self._is_prior_param(name)
            or self._is_post_param(name)
            or self._is_dist_param(name)
            or name.startswith("param_")
            or name.startswith("bias_cond")
            or name.startswith("embed_to_dynamic")
            or name.startswith("dynamic_to_cond_scale.")
        )
    
    def get_updatable_params(self):
        for n, p in self.named_parameters():
            if self._is_updatable_param(n):
                yield n, p

    def get_posterior_params(self):
        print("latent param:")
        for n, p in self.get_updatable_params():
            if self._is_latent_param(n):
                print(n)
                yield p

    def get_model_params(self):
        print("model param:")
        for n, p in self.get_updatable_params():
            if not self._is_latent_param(n):
                print(n)
                yield p

    def reset(self):
        self.cond_sample = None
        self.last_age = None

    def update_posterior(self):
        for n, p in self.get_updatable_params():
            if self._is_latent_param(n):
                p.requires_grad = True
            else:
                p.requires_grad = False
        self.updating_posterior = True
        self.updating_model = False
        self.reset()
    
    def update_model(self):
        for n, p in self.get_updatable_params():
            if self._is_latent_param(n):
                p.requires_grad = False
            else:
                p.requires_grad = True
        self.updating_posterior = False
        self.updating_model = True
        self.reset()
        self.set_dim_idx(self.get_full_dim_idx())

    def eval(self):
        super().eval()
        self.updating_posterior = False
        self.updating_model = False
        self.reset()
    
    def get_init_sample(self, batch_size, dim_idx):
        return self.prior.get_init_dist(batch_size * self.n_sample_param, dim_idx=dim_idx).sample()

    def update_sample(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        requires_grad=False,
        last_only=False,
    ):
        full_dim_idx = self.get_full_dim_idx()
        samples = []
        with torch.set_grad_enabled(requires_grad):
            for dim_idx in torch.split(full_dim_idx, self.n_sample_dim):
                len_pred = self.len_infer_post
                cond_sample = self.get_init_sample(past_feat_dynamic_age.shape[0], dim_idx=dim_idx)
                sample, dists_q, prob_q = self.sample_param_posterior_var(
                    past_target,
                    past_feat_dynamic_age,
                    cond_sample=cond_sample,
                    start=-len_pred,
                    dim_idx=dim_idx,
                )
                samples.append(sample)
        param_sample = torch.cat(samples, dim=-2)
        param_sample = self.expand_param_sample(param_sample, len_pred)
        self.cond_sample = param_sample
    
    def update_sample_rolling(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        requires_grad=False,
        last_only=False,
    ):
        len_pred = self.len_infer_post
        if self.cond_sample is None:
            cond_sample = self.get_init_sample(past_feat_dynamic_age.shape[0], dim_idx=self.dim_idx)
        else:
            cond_sample = self.cond_sample[:, -1:]
            len_diff = (past_feat_dynamic_age.max(dim=1)[0] - self.last_age).squeeze().long()
            if self.len_infer_post > len_diff:
                # last batch
                len_pred = len_diff
        self.last_age = past_feat_dynamic_age.max(dim=1)[0].detach()
        with torch.set_grad_enabled(requires_grad):
            param_sample, dists_q, prob_q = self.sample_param_posterior_var(
                past_target,
                past_feat_dynamic_age,
                cond_sample=cond_sample,
                start=-len_pred,
                dim_idx=self.dim_idx,
            )
            param_sample = self.expand_param_sample(param_sample, len_pred)
            if last_only:
                self.cond_sample = param_sample[:, -1:]
            else:
                if self.cond_sample is None:
                    self.cond_sample = param_sample
                else:
                    self.cond_sample = torch.cat(
                        (self.cond_sample, param_sample),
                        dim=1,
                    )
                assert (self.last_age == self.cond_sample.shape[1] - 1 + self.len_hist).all()
    
    def expand_param_sample(self, param_sample, len_total):
        if self.step_param != 1:
            param_sample = torch.repeat_interleave(param_sample, self.step_param, 1)
            param_sample = param_sample[:, :len_total]
        return param_sample

    def get_full_dim_idx(self):
        return torch.arange(self.dim_target, device=self.param_shift.device)

    def set_dim_idx(self, dim_idx):
        self.dim_idx = dim_idx

    def get_len_pred(self):
        if self.updating_model:
            return self.len_infer_model
        else:
            return self.len_infer_post

    def training_forward_1(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
    ):
        past_target, log_det_jac = self.transform_target(past_target, past_time_feat)
        embedding = self.get_embedding(
            past_target,
            past_feat_dynamic_age,
            past_time_feat,
            past_observed_values,
            past_is_pad,
            target_dimension_indicator,
            len_pred=self.get_len_pred(),
        )
        return past_target, log_det_jac, embedding
    
    def training_forward_2(
        self,
        past_target_original,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        past_target,
        log_det_jac,
        embedding,
    ):
        dim_idx = self.dim_idx
        len_pred = self.get_len_pred()
        if self.updating_model:
            # train model
            assert self.cond_sample is not None
            # (N * S, T, C)
            cond_sample = self.select_samples(
                self.cond_sample,
                past_feat_dynamic_age[:, -len_pred:, 0].long(),
            )
            param_sample = cond_sample
            KL = None
        else: # updating_posterior
            # train posterior
            # first time point
            if self.cond_sample is None:
                assert (past_feat_dynamic_age.min(dim=1)[0] == 0.0).all()
                cond_sample = self.get_init_sample(past_feat_dynamic_age.shape[0], dim_idx)
            else:
                len_diff = (past_feat_dynamic_age.max(dim=1)[0] - self.last_age).squeeze().long()
                if len_pred > len_diff:
                    len_pred = len_diff
                cond_sample = self.cond_sample[:, -1:]
            param_sample, dists_q, prob_q = self.sample_param_posterior_var(
                past_target,
                past_feat_dynamic_age,
                cond_sample=cond_sample,
                start=-len_pred,
                dim_idx=dim_idx,
            )
            self.cond_sample = param_sample[:, -1:]
            self.last_age = past_feat_dynamic_age.max(dim=1)[0].detach()
        if self.updating_posterior:
            dists_p, prob_p = self.prior(param_sample, True, cond_sample, dim_idx=dim_idx)
            if len(prob_p.shape) == len(prob_q.shape) - 1:
                # same time length
                assert prob_p.shape[1] == prob_q.shape[1]
                # sum over dims
                prob_q = prob_q.sum((2, 3))
                prob_p = prob_p.sum(2)
            else:
                prob_q = prob_q.sum(2)
                prob_p = prob_p.sum(2)
            KL = prob_q - prob_p
            param_sample = self.expand_param_sample(param_sample, len_pred)
        # target shape (N, T, C)
        # sample shape (S * N, T, C)
        # create input and output
        cond = self.embedding_to_encoding(embedding[0], embedding[1], param_sample, dim_idx)
        y_out = past_target[:, -len_pred:, dim_idx]
        y_out, log_det_jac_out = self.transform_output(y_out, dim_idx, self.dequantize)
        LL = self.dist.log_prob(
            y_out.unsqueeze(1).expand(-1, cond.shape[1], -1, -1),
            cond,
            dim_idx=dim_idx,
            cond_scale=self.cond_scale(embedding[1], dim_idx),
        )
        LL = self.transform_loglike(LL, log_det_jac, log_det_jac_out, dim_idx, len_pred, cond.shape[1])
        return self.compute_loss(LL, KL)

    def training_forward(self, *args):
        output = self.training_forward_1(*args)
        return self.training_forward_2(*args, *output)

    def compute_loss(self, LL, KL):
        if KL is None:
            # avg over batch size, posterior sample size, prediction length
            return (-LL.mean(),)
        else:
            KL = KL.mean()
            NLL = -LL.mean()
            loss = NLL + KL
            return loss, NLL, KL

    def get_embedding(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        len_pred,
    ):
        full_time_idx = torch.arange(len_pred, device=past_target.device)
        if self.n_sample_time:
            time_idxs = torch.split(full_time_idx, self.n_sample_time)
            cache = {} 
        else:
            time_idxs = [full_time_idx]
            cache = None
        h_list = []
        x_p_list = []
        x_roll = self.create_input(
            past_target=past_target,
            past_time_feat=past_time_feat,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            target_dimension_indicator=target_dimension_indicator,
            len_pred=len_pred,
            cache=cache,
        )
        for time_idx in time_idxs:
            # (N, T, C, L) -> (N, T, L, C)
            x_in = x_roll[:, time_idx, :-1]
            x_p = x_roll[:, time_idx, -1:]
            h = self.x_to_h(x_in, x_p)
            h_list.append(h)
            x_p_list.append(x_p)
        h = torch.cat(h_list, dim=1)
        x_p = torch.cat(x_p_list, dim=1)
        return h, x_p

    def embedding_to_encoding(self, h, x_p, param, dim_idx):
        N = h.shape[0]
        T = h.shape[1]
        h = self.embed_to_dynamic(h)
        # stochastic part
        eta_w = self.transform_param(param, x_p, dim_idx)
        # M = B * S
        # N = B or N = B * S
        eta_w = eta_w.reshape(N, -1, T, *eta_w.shape[2:])
        h = h[:, None, :, :].expand(-1, eta_w.shape[1], -1, -1)
        dynamic = (h.reshape(eta_w.shape[0], eta_w.shape[1], eta_w.shape[2], -1, eta_w.shape[4])[:, :, :, dim_idx, :] * eta_w).sum(-1)
        cond_dynamic = self.dynamic_to_cond(dynamic, dim_idx)
        return cond_dynamic

    def x_to_h(self, x, x_p):
        """
        x: (N, T, L, C)
        """
        N = x.shape[0]
        T = x.shape[1]
        assert x.shape[2] == self.len_context
        h = self.encoder(
            x.reshape(N * x.shape[1], x.shape[2], x.shape[3]),
            x_p.reshape(N * x_p.shape[1], x_p.shape[2], x_p.shape[3])
        )
        h = h.reshape(N, T, h.shape[-1])
        return h

    def get_encoding(self, param, x, x_p, dim_idx):
        h = self.x_to_h(x, x_p)
        return self.embedding_to_encoding(h, x_p, param, dim_idx)
    
    def cond_scale(self, x_p, dim_idx):
        # add dim for samples
        x_p = x_p[:, None]
        return super().cond_scale(x_p, dim_idx)

    def sample_param_posterior_pf(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator=None,
        last_only=True,
        dim_idx=None,
    ):
        S = self.n_sample_param
        N = past_target.shape[0]
        H = self.len_hist
        L = self.len_context
        # prior sample for t = 0
        init_sample = self.get_init_sample(N, dim_idx)
        sample = init_sample
        T, O, I = sample.shape[1:]
        post_samples = []
        def pre_get_log_like(t):
            # include current t to create future feature
            x = self.create_input(
                past_target=past_target[:, t-H:t+1],
                past_time_feat=past_time_feat[:, t-H:t+1],
                past_observed_values=past_observed_values[:, t-H:t+1],
                past_is_pad=past_is_pad[:, t-H:t+1],
                target_dimension_indicator=target_dimension_indicator,
                len_pred=1,
            )
            x_in = x[:, :, :-1]
            x_p = x[:, :, -1:]
            y_out = past_target[:, t:t+1, dim_idx]
            y_out, log_det_jac_out = self.transform_output(y_out, dim_idx, dequantize=False)
            return x_in, x_p, y_out
        def get_log_like(sample, x_in, x_p, y_out):
            cond = self.get_encoding(
                sample,
                x_in,
                x_p,
                dim_idx,
            )
            log_like = self.dist.log_prob(
                y_out.unsqueeze(1).expand(-1, S, -1, -1),
                cond,
                dim_idx=dim_idx,
                cond_scale=self.cond_scale(x_p, dim_idx),
            )
            return log_like
        def get_log_like_full(sample, t):
            x_in, x_p, y_out = pre_get_log_like(t)
            log_like = get_log_like(sample, x_in=x_in, x_p=x_p, y_out=y_out)
            return log_like
        log_w = 0.0
        # sequential sampling
        for t in range(H, past_target.shape[1]):
            prev_sample = sample
            sample = self.prior.get_next_dist(prev_sample, dim_idx).sample()
            log_like = get_log_like_full(sample, t)
            # (N, S, T, C) -> (N, T, C, S)
            log_w = log_like.permute(0, 2, 3, 1)
            dist_w = Categorical(logits=log_w)
            # normalize log weights
            # (S, N, T, C) -> (N, S, T, C)
            idx = dist_w.sample(torch.Size([S])).permute(1, 0, 2, 3)
            sample_re = sample.reshape(-1, S, T, O, I).gather(1, idx[:, :, :, :, None].expand(-1, -1, -1, -1, I)).reshape(-1, T, O, I)
            post_samples.append(sample_re)
            sample = post_samples[-1]
        if last_only:
            post_sample = sample
        else:
            post_sample = torch.cat(post_samples, dim=1)
        return post_sample, None, None

    def sample_param_posterior_rbpf(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator=None,
        last_only=True,
        dim_idx=None,
    ):
        S = self.n_sample_param
        N = past_target.shape[0]
        H = self.len_hist
        L = self.len_context
        T = past_target.shape[1]
        n_steps = (T - H) // self.step_param
        T = n_steps * self.step_param + H
        past_target = past_target[:, -T:]
        past_time_feat = past_time_feat[:, -T:]
        past_observed_values = past_observed_values[:, -T:]
        past_is_pad = past_is_pad[:, -T:]
        def pre_get_log_like(t, len_pred):
            # include current t to create future feature
            x = self.create_input(
                past_target=past_target[:, t-H:t+len_pred],
                past_time_feat=past_time_feat[:, t-H:t+len_pred],
                past_observed_values=past_observed_values[:, t-H:t+len_pred],
                past_is_pad=past_is_pad[:, t-H:t+len_pred],
                target_dimension_indicator=target_dimension_indicator,
                len_pred=len_pred,
            )
            x_in = x[:, :, :-1]
            x_p = x[:, :, -1:]
            y_out = past_target[:, t:t+len_pred, dim_idx]
            y_out, log_det_jac_out = self.transform_output(y_out, dim_idx, dequantize=False)
            return x_in, x_p, y_out
        # prior sample for t = 0
        init_dist = self.prior.get_init_dist(N * self.n_sample_param, dim_idx=dim_idx)
        T, O, I = init_dist.sample().shape[1:]
        m_0 = self.prior.mu_0.expand(dim_idx.shape[-1], self.dim_param)
        s_d = torch.diag_embed(self.prior.get_scale_d(dim_idx)**2)
        s_0 = torch.diag_embed(self.prior.get_scale_0(dim_idx)**2)
        lambd = self.prior.get_lambd(dim_idx)
        mix_dist = Categorical(
            torch.stack((lambd, 1.0-lambd), dim=-1)
        )
        def get_posterior(pi, m, P, x, x_p, y_out):
            N = x.shape[0]
            T = x.shape[1]
            assert x.shape[2] == self.len_context
            assert T == 1
            h = self.encoder(
                x.reshape(N * x.shape[1], x.shape[2], x.shape[3]),
                x_p.reshape(N * x_p.shape[1], x_p.shape[2], x_p.shape[3])
            )
            if pi is not None:
                m = torch.where(pi[:, :, :, :, None] == 0, m_0, m)
                P = torch.where(pi[:, :, :, :, None, None] == 0, s_0, P + s_d)
            h = h.reshape(N, T, h.shape[-1])
            # (N, T, I * O)
            h = self.embed_to_dynamic(h)
            eta_w_shift = self.param_shift[dim_idx]
            eta_w_scale = self.param_scale(x_p, dim_idx)
            # (N, S, T, O * I)
            h = h[:, None, :, :].expand(-1, m.shape[1], -1, -1)
            # (N, S, T, O, I)
            h = h.reshape(h.shape[0], h.shape[1], h.shape[2], -1, m.shape[4])[:, :, :, dim_idx, :] 
            static = (h * eta_w_shift).sum(-1)
            bias = self.bias_cond[dim_idx]
            y_out = y_out.unsqueeze(1)
            if self.dist.scale is not None:
                y_out = y_out / self.dist.scale[..., dim_idx]
            y = y_out.expand(-1, m.shape[1], -1, -1) - bias - static
            h = h * eta_w_scale.unsqueeze(1)
            Hm = (h * m).sum(-1)
            z = (y - Hm)[:, :, :, :, None, None]
            R = self.dist.get_sigma(dim_idx, cond_scale=self.cond_scale(x_p, dim_idx))**2
            # each y is one dim, so add (1, 1) to dim
            R = R[..., None, None]
            H = h.unsqueeze(-2)
            PHt = P @ H.transpose(-2, -1)
            # S is one dim
            S = H @ PHt + R
            K = PHt / S
            m = m + (K @ z).squeeze(-1)
            I = torch.eye(P.shape[-1], device=P.device)
            IKH = (I - K @ H)
            P = IKH @ P @ IKH.transpose(-2, -1) + R * K @ K.transpose(-2, -1)
            P = (P + P.transpose(-2, -1)) / 2
            marginal = Normal(Hm, S.view(*S.shape[:-2])).log_prob(y)
            return marginal, m, P
        post_samples = []
        m_t = m_0
        s_t = s_0
        for t in range(H, past_target.shape[1], self.step_param):
            pi_sample = mix_dist.sample(torch.Size([N, S, 1]))
            x_in, x_p, y_out = pre_get_log_like(t, 1)
            log_like, m_t_cond, s_t_cond = get_posterior(pi_sample, m_t, s_t, x_in, x_p, y_out)
            for i in range(1, self.step_param):
                x_in, x_p, y_out = pre_get_log_like(t+i, 1)
                log_like_i, m_t_cond, s_t_cond = get_posterior(None, m_t_cond, s_t_cond, x_in, x_p, y_out)
                log_like += log_like_i
            # (N, S, T, C) -> (N, T, C, S)
            log_w = log_like.permute(0, 2, 3, 1)
            dist_w = Categorical(logits=log_w)
            # normalize log weights
            # (S, N, T, C) -> (N, S, T, C)
            idx = dist_w.sample(torch.Size([S])).permute(1, 0, 2, 3)
            m_t = m_t_cond.gather(1, idx[:, :, :, :, None].expand(-1, -1, -1, -1, I))
            s_t = s_t_cond.gather(1, idx[:, :, :, :, None, None].expand(-1, -1, -1, -1, I, I))
            sample = MultivariateNormal(m_t, s_t).sample().reshape(-1, T, O, I)
            if not last_only:
                post_samples.append(sample)
        if last_only:
            post_sample = sample
        else:
            post_sample = torch.cat(post_samples, dim=1)
        return post_sample, None, None

    def eval_forward(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
    ):
        full_dim_idx = torch.arange(self.dim_target, device=past_target.device)
        loss = 0.0
        dim_idx = full_dim_idx
        past_target, log_det_jac = self.transform_target(past_target, past_time_feat)
        len_pred = self.len_pred
        post_param_sample, _, _ = self.sample_param_posterior_rbpf(
            past_target=past_target[:, :-len_pred],
            past_feat_dynamic_age=past_feat_dynamic_age[:, :-len_pred],
            past_time_feat=past_time_feat[:, :-len_pred],
            past_observed_values=past_observed_values[:, :-len_pred],
            past_is_pad=past_is_pad[:, :-len_pred],
            target_dimension_indicator=target_dimension_indicator,
            last_only=True,
            dim_idx=dim_idx,
        )
        post_param_sample = post_param_sample[:, -1:]
        param_sample = self.prior.sample(post_param_sample, len_pred, dim_idx)
        L = self.len_context
        x_roll = self.create_input(
            past_target=past_target,
            past_time_feat=past_time_feat,
            past_observed_values=past_observed_values,
            past_is_pad=past_is_pad,
            target_dimension_indicator=target_dimension_indicator,
            len_pred=len_pred,
        )
        # (N, T, C, L) -> (N, T, L, C)
        x_in = x_roll[:, :, :-1]
        x_p = x_roll[:, :, -1:]
        cond = self.get_encoding(param_sample, x_in, x_p, dim_idx)
        y_out = past_target[:, -len_pred:, dim_idx]
        y_out, log_det_jac_out = self.transform_output(y_out, dim_idx, dequantize=False)
        # (N, T, C) -> (N, S, T, C) -> (N, S, T)
        LL = self.dist.log_prob(
            y_out.unsqueeze(1).expand(-1, cond.shape[1], -1, -1),
            cond,
            dim_idx=dim_idx,
            cond_scale=self.cond_scale(x_p, dim_idx)
        )
        LL = self.transform_loglike(LL, log_det_jac, log_det_jac_out, dim_idx, len_pred, cond.shape[1])
        log_marginal = torch.logsumexp(LL, 1) - np.log(LL.shape[1])
        loss = loss + (-log_marginal).mean(0).sum(0)
        return (loss,)


class NNARPredictionNetwork(NNARTrainingNetwork):
    def __init__(
        self,
        n_sample_target,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_sample_target = n_sample_target
        self.n_sample_param = n_sample_target
        self.posterior.n_sample = n_sample_target
        self.use_var_posterior = False

    def sample_param_prior(
        self,
        past_target,
        past_feat_dynamic_age=None,
        past_time_feat=None,
        past_observed_values=None,
        past_is_pad=None,
        target_dimension_indicator=None,
        dim_idx=None,
    ):
        if self.use_var_posterior:
            past_feat_dynamic_age = past_feat_dynamic_age + self.len_pred
            post_param_sample, _, _ = self.sample_param_posterior_var(
                past_target,
                past_feat_dynamic_age,
                past_time_feat,
                past_observed_values,
                past_is_pad,
                target_dimension_indicator,
                last_only=False,
                start=-self.len_pred,
                dim_idx=dim_idx,
                cond_sample=self.get_init_sample(past_feat_dynamic_age.shape[0], dim_idx),
            )
            param_sample = post_param_sample[:, -self.len_pred:]
        else:
            post_param_sample, _, _ = self.sample_param_posterior_rbpf(
                past_target,
                past_feat_dynamic_age,
                past_time_feat,
                past_observed_values,
                past_is_pad,
                target_dimension_indicator,
                last_only=True,
                dim_idx=dim_idx,
            )
            # only need last time point
            param_sample = self.prior.sample(post_param_sample[:, -1:], math.ceil(self.len_pred / self.step_param), dim_idx)
            param_sample = self.expand_param_sample(param_sample, self.len_pred)
        return param_sample

    def forward(
        self,
        past_target,
        past_feat_dynamic_age,
        past_time_feat,
        past_observed_values,
        past_is_pad,
        target_dimension_indicator,
        future_time_feat,
    ):
        past_target, _ = self.transform_target(past_target, past_time_feat)
        dim_idx = torch.arange(self.dim_target, device=past_target.device)
        param_sample = self.sample_param_prior(
            past_target,
            past_feat_dynamic_age,
            past_time_feat,
            past_observed_values,
            past_is_pad,
            target_dimension_indicator,
            dim_idx=dim_idx,
        )
        H = self.len_hist
        past_target = past_target[:, -H:]
        past_time_feat = past_time_feat[:, -H:]
        past_observed_values = past_observed_values[:, -H:]
        assert (past_observed_values == 1.0).all()
        past_is_pad = past_is_pad[:, -H:]
        assert (past_is_pad == 0.0).all()
        y = torch.repeat_interleave(past_target, self.n_sample_param, dim=0)
        time_feat = torch.repeat_interleave(past_time_feat, self.n_sample_param, dim=0)
        future_time_feat = torch.repeat_interleave(future_time_feat, self.n_sample_param, dim=0)
        target_dimension_indicator = torch.repeat_interleave(target_dimension_indicator, self.n_sample_param, dim=0)
        y_out = []
        y_pad = torch.zeros_like(y[:, -1:])
        for t in range(self.len_pred):
            time_feat = torch.cat((time_feat[:, 1:], future_time_feat[:, t:t+1]), dim=1)
            past_target_t = torch.cat((y, y_pad), dim=1)
            x_roll = self.create_input(
                past_target=past_target_t,
                past_time_feat=time_feat,
                past_observed_values=torch.ones_like(past_target_t),
                past_is_pad=torch.zeros(*past_target_t.shape[:-1], device=past_target_t.device),
                target_dimension_indicator=target_dimension_indicator,
                len_pred=1,
            )
            x_in = x_roll[:, :, :-1]
            x_p = x_roll[:, :, -1:]
            # add len_pred = 1 as dim=1
            cond = self.get_encoding(
                param_sample[:, t:t+1],
                x_in,
                x_p,
                dim_idx=dim_idx,
            )
            # (B*S, 1, T, C)
            y_out_t, _ = self.inverse_transform_output(
                self.dist.sample(dim_idx=dim_idx, cond=cond, cond_scale=self.cond_scale(x_p, dim_idx)).squeeze(1),
                dim_idx,
            )
            y_out.append(y_out_t)
            # be careful when len_context == 1
            y = torch.cat((y[:, 1:], y_out[-1]), dim=1)
        y_out = torch.cat(y_out, dim=1)
        y_out, _ = self.inverse_transform_target(y_out, future_time_feat)
        if self.dim_target == 1:
            # univariate prediction
            return y_out.reshape(
                -1,
                self.n_sample_target,
                self.len_pred,
            )
        else:
            return y_out.reshape(
                -1,
                self.n_sample_target,
                self.len_pred,
                self.dim_target,
            )

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.distributions import Normal

from ..constants import PLOT_FOLDER, DATA_FOLDER


class ARModel(nn.Module):
    def __init__(self, lag, len_total, coeff, mu, sigma):
        super().__init__()
        self.lag = lag
        self.len_total = len_total
        self._set_param("coeff", coeff, 2)
        self._set_param("mu", mu, 1)
        self._set_param("sigma", sigma, 1)
    
    def _set_param(self, name, value, ndim, dim=0):
        value = torch.tensor(value).float()
        while len(value.shape) < ndim:
            value = value.unsqueeze(0)
        value = torch.repeat_interleave(value, self.len_total // value.shape[dim], dim=dim)
        self.register_buffer(name, value)
    
    def forward(self, y, t):
        mu = self.mu[t] + (self.coeff[t] * y[..., -self.lag:]).sum(-1)
        return Normal(mu, self.sigma[t])
    
    def sample(self, y, t):
        return self(y, t).sample()


class VARModel(nn.Module):
    """
    only supports order = 1
    """
    def __init__(self, len_total, coeff):
        super().__init__()
        self.len_total = len_total
        self._set_param("coeff", coeff, 3)
    
    def _set_param(self, name, value, ndim, dim=0):
        value = torch.FloatTensor(value)
        while len(value.shape) < ndim:
            value = value.unsqueeze(0)
        value = torch.repeat_interleave(value, self.len_total // value.shape[dim], dim=dim)
        self.register_buffer(name, value)

    def forward(self, y, t):
        mu = y[..., -1, :].matmul(self.coeff[t])
        return Normal(mu, 1.0)
    
    def sample(self, y, t):
        return self(y, t).sample()


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def generate_ar(
        len_total,
        lag,
        coeff,
        mu,
        sigma,
        y_c=None,
        model_cls=ARModel,
    ):
    if y_c is None: 
        y_c = np.random.normal(loc=0.0, scale=1.0, size=lag)
    y = torch.FloatTensor(y_c)
    model = model_cls(lag=lag, coeff=coeff, mu=mu, sigma=sigma, len_total=len_total)
    for t in range(len_total):
        y = torch.cat((y, model.sample(y, t).unsqueeze(0)), dim=0)
    y = y[len(y_c):]
    print(coeff)
    return y, {
        "lag": lag,
        "y_c": y_c,
        "coeff": coeff,
        "mu": mu,
        "sigma": sigma,
        "len_total": len_total,
    }


def simulate_dynamic_ar(
        len_total,
        n_per_regime=100,
        n_lag=1,
        lag_choices=list(range(1, 5+1))+list(range(10, 50+1, 10)),
        coeff_choices=None,
        mu=0.0,
        sigma=1.0,
        seed=None,
    ):
    if seed is not None:
        set_seed(seed)
    assert (len_total % n_per_regime == 0)
    n_regime = len_total // n_per_regime
    lag_max = max(lag_choices)
    lag_choices = np.array(lag_choices)
    coeffs = []
    for _ in range(n_regime):
        # sample lags
        lags = np.random.choice(lag_choices, n_lag, replace=False).tolist()
        coeff = np.zeros(lag_max)
        for l in lags:
            if coeff_choices:
                coeff[l-1] = np.random.choice(coeff_choices)
            else:
                coeff[l-1] = np.random.uniform(-1, 1)
        coeffs.append(coeff)
    y, model = generate_ar(len_total, lag_max, coeffs, mu, sigma)
    model.update({
        "t_regime": [i*n_per_regime for i in range(n_regime)],
    })
    return y, model


def simulate_sin_ar(
        len_total=2000,
        mu=0.0,
        sigma=1.0,
        coeff_bound=0.8,
        seed=None,
    ):
    if seed is not None:
        set_seed(seed)
    time = np.arange(len_total)
    # coeff needs to be 2 D, where first dim is time
    coeffs = (coeff_bound * np.sin(time / len_total * 2 * np.pi))[:, np.newaxis]
    y, model = generate_ar(len_total, 1, coeffs, mu, sigma)
    return y, model


def simulate_ar(
        len_total=2000,
        params=[0.5],
        sigma=1.0,
        seed=None,
    ):
    if seed is not None:
        set_seed(seed)
    lag = len(params)
    w = np.array(params)
    y, model = generate_ar(len_total, lag, w, 0.0, sigma)
    return y, model


def generate_var(
        len_total,
        coeff,
        y_c=None,
    ):
    dim = coeff[0].shape[-1]
    if y_c is None: 
        y_c = np.random.normal(loc=0.0, scale=1.0, size=(1, dim))
    y = torch.FloatTensor(y_c)
    model = VARModel(coeff=coeff, len_total=len_total)
    for t in range(len_total):
        y = torch.cat((y, model.sample(y, t)[None, :]), dim=0)
    y = y[len(y_c):].numpy()
    print(coeff)
    return y, {
        "y_c": y_c,
        "coeff": coeff,
        "len_total": len_total,
    }


def simulate_dynamic_var(
        len_total=2000,
        n_per_regime=100,
        dim=4,
        seed=None,
    ):
    if seed is not None:
        set_seed(seed)
    assert (len_total % n_per_regime == 0)
    n_regime = len_total // n_per_regime
    coeffs = []
    for _ in range(n_regime):
        max_abs = np.inf
        while max_abs >= 1:
            coeff = np.random.uniform(low=-0.8, high=0.8, size=(dim, dim))
            eigval, _ = np.linalg.eig(coeff)
            max_abs = np.max(np.abs(eigval))
        print(max_abs)
        coeffs.append(coeff)
    y, model = generate_var(len_total, coeffs)
    model.update({
        "t_regime": [i*n_per_regime for i in range(n_regime)],
    })
    return y, model


def plot_dynamic_ar(y, params):
    plt.figure()
    plt.plot(y)
    t_regime = params["t_regime"]
    n_regime = len(t_regime)
    if n_regime > 1:
        for x in t_regime:
            plt.axvline(x=x, color='k', linestyle='--')


def plot_ts(y, params):
    plt.figure()
    plt.plot(y)


def run_simulation(name, sim, plot):
    y, params = sim
    figpath = PLOT_FOLDER / f"{name}.png"
    PLOT_FOLDER.mkdir(parents=True, exist_ok=True)
    plot(y, params)
    plt.savefig(figpath)
    folder = DATA_FOLDER / name
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "model.pkl", "wb") as f:
        pickle.dump(params, f)
    if len(y.shape) < 2 or y.shape[-1] == 1:
        columns = ["y"]
    else:
        columns = [f"y_{i}" for i in range(y.shape[-1])]
    df = pd.DataFrame(y, index=range(1, len(y)+1), columns=columns)
    print(df)
    df.to_pickle(folder / "data.pkl")

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from gluonts.transform import TransformedDataset
from gluonts.dataset.field_names import FieldName

from dcf.model.nnar import scale_transform


# adpated from GluonTS tutorial
def plot_prob_forecasts(target, forecast, plot_length, prediction_intervals=(50.0, 90.0, 95.0, 99.0)):
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    target[-plot_length:].plot(ax=ax)
    forecast.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")


def plot_prob_forecasts_multivariate(
        target,
        forecast,
        plot_length,
        dim,
        prediction_intervals=(50.0, 90.0),
        label_prefix="",
        color='g',
    ):
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    ps = [50.0] + [
            50.0 + f * c / 2.0 for c in prediction_intervals for f in [-1.0, +1.0]
        ]
        
    percentiles_sorted = sorted(set(ps))
    
    def alpha_for_percentile(p):
        return (p / 100.0) ** 0.3
        
    target[-plot_length:][dim].plot(ax=ax)
    
    ps_data = [forecast.quantile(p / 100.0)[:,dim] for p in percentiles_sorted]
    i_p50 = len(percentiles_sorted) // 2
    
    p50_data = ps_data[i_p50]
    p50_series = pd.Series(data=p50_data, index=forecast.index)
    p50_series.plot(color=color, ls="-", label=f"{label_prefix}median", ax=ax)
    
    for i in range(len(percentiles_sorted) // 2):
        ptile = percentiles_sorted[i]
        alpha = alpha_for_percentile(ptile)
        ax.fill_between(
            forecast.index,
            ps_data[i],
            ps_data[-i - 1],
            facecolor=color,
            alpha=alpha,
            interpolate=True,
        )
        pd.Series(data=p50_data[:1], index=forecast.index[:1]).plot(
            color=color,
            alpha=alpha,
            linewidth=10,
            label=f"{label_prefix}{100 - ptile * 2}%",
            ax=ax,
        )

    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]    
    ax.legend(legend, loc="upper left")


def plot_latent(estimator, net, dss, base_fig_path, use_datetime=False, ds_params=None):
    len_hist = estimator.len_hist
    device = "cpu"
    net.to(device)
    dim_idx = torch.arange(net.dim_target, device=device)
    with torch.no_grad():
        for n, p in net.prior.named_parameters():
            print(n)
            print(p)
            if n == "log_sigma_d" and hasattr(net.prior, "get_scale_d"):
                print(net.prior.get_scale_d(dim_idx))
            elif n.startswith("log"):
                print(f"scale: {scale_transform(p)}")
                print(f"sigmoid: {torch.sigmoid(p)}")
            elif n.startswith("mix"):
                print(f"softmax: {torch.softmax(p, dim=-1)}")
        print(f"lambd: {net.prior.get_lambd(dim_idx)}")
    if ds_params:
        print(ds_params)
        t_regime = ds_params.get("t_regime")
        if t_regime:
            n_regime = len(t_regime)
        else:
            n_regime = 1
    else:
        n_regime = 1
    for ds_split, ds in dss.items():
        if len(ds) > 1:
            print("Dataset contains multiple series. Use the longest one only for plotting.")
        transformation = estimator.create_transformation()
        ds = TransformedDataset(
            ds,
            transformation,
            is_train=False,
        )
        data = max(list(ds), key=lambda d: d["target"].shape[1])
        start = data[FieldName.START]
        target = data[FieldName.TARGET].T
        len_total = target.shape[0]
        age = (data["feat_dynamic_age"].T)[:len_total]
        time_feat = (data["time_feat"].T)[:len_total]
        target_dim_indicator = data["target_dimension_indicator"]
        target = torch.tensor(target)[None, :, :].to(device)
        age = torch.tensor(age)[None, :, :].to(device)
        time_feat = torch.tensor(time_feat)[None, :, :].to(device)
        observed_values = torch.ones_like(target)
        is_pad = torch.zeros_like(age).squeeze(-1)
        target_dim_indicator = torch.tensor(target_dim_indicator)[None, :].to(device)
        target, _ = net.transform_target(target, None)
        def as_inference_var():
            param_samples = []
            time = np.arange(len_hist, len_total)
            with torch.no_grad():
                param_sample = net.expand_param_sample(net.sample_param_posterior_var(
                    target,
                    age,
                    time_feat,
                    observed_values,
                    is_pad,
                    target_dimension_indicator=target_dim_indicator,
                    dim_idx=dim_idx,
                    start=len_hist,
                )[0], len(time))
                param_samples.append(param_sample)
            return "inference_var", param_samples, time
        def as_inference_rbpf():
            param_samples = []
            time = np.arange(len_hist, len_total)
            with torch.no_grad():
                param_sample = net.expand_param_sample(net.sample_param_posterior_rbpf(
                    target,
                    age,
                    time_feat,
                    observed_values,
                    is_pad,
                    target_dimension_indicator=target_dim_indicator,
                    last_only=False,
                    dim_idx=dim_idx,
                )[0], len(time))
                if param_sample.shape[1] < len(time):
                    time = time[-param_sample.shape[1]:]
                param_samples.append(param_sample)
            return "inference_rbpf", param_samples, time
        def plot_coeff(j, k, ax):
            if ds_params:
                coeff = ds_params["coeff"]
                T = ds_params["len_total"]
                t_regime = ds_params.get("t_regime")
                style = {
                    "color": "r",
                    "linestyle": "--",
                }
                if t_regime:
                    ts = t_regime + [T]
                    for i in range(len(ts) - 1):
                        if len(coeff[i].shape) == 1:
                            ax.hlines(y=coeff[i], xmin=ts[i], xmax=ts[i+1], **style)
                        else:
                            ax.hlines(y=coeff[i][k, j], xmin=ts[i], xmax=ts[i+1], **style)
                else:
                    if len(coeff) > 1:
                        ax.plot(coeff, **style)
                    else:
                        ax.hlines(y=coeff[0], x_min=0, x_max=T, **style)
        def plot_regime(ax):
            if n_regime > 1:
                for x in t_regime:
                    ax.axvline(x=x, color='k', linestyle='--', alpha=.3)
        def plot_quantile(param_sample, time, j, k):
            fig, ax = plt.subplots(figsize=(6, 4))
            plot_regime(ax)
            median = np.median(param_sample, axis=0)
            quantiles = np.quantile(param_sample, (0.05, 0.95), axis=0)
            # drop first few points
            len_drop = 10
            time = time[len_drop:]
            median = median[len_drop:]
            quantiles = quantiles[:, len_drop:]
            if use_datetime:
                dt = [start + t * start.freq for t in time]
                median_series = pd.Series(median, dt)
                median_series.plot(color="b")
                ax.fill_between(dt, quantiles[0], quantiles[1], color='b', alpha=.3)
            else:
                plt.plot(time, median, color="b")
                ax.fill_between(time, quantiles[0], quantiles[1], color='b', alpha=.3)
            plot_coeff(j, k, ax=ax)
            return fig
        if ds_split == "train":
            methods = [
                as_inference_var,
                as_inference_rbpf,
            ]
        else:
            methods = [
                as_inference_rbpf,
            ]
        plot_funcs = {
            "quantile": plot_quantile
        }
        plt.rcParams.update({'font.size': 10})
        for method in methods:
            name, param_samples, time = method()
            if len(param_samples) > 0:
                with torch.no_grad():
                    param_sample = torch.cat(param_samples, dim=1)
                for fig_type, plot_func in plot_funcs.items():
                    fig_path = base_fig_path / name / fig_type
                    fig_path.mkdir(parents=True, exist_ok=True)
                    for ptype, sample in {"latent": param_sample}.items():
                        sample = sample.cpu().numpy()
                        for j in range(min(sample.shape[-2], 100)):
                            for k in range(min(sample.shape[-1], 10)):
                                plot_func(sample[..., j, k], time, j, k)
                                plt.savefig(fig_path / f"{ds_split}_{ptype}_{j}_{k}.png", bbox_inches='tight', dpi=300)
                                plt.close()

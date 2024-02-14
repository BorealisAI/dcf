# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from pathlib import Path

import numpy as np
import pandas as pd

from dcf.model.nnar import NNAREstimator, NAREstimator
from dcf.model.nnar import (
    ShiftScaleLayer,
    ShiftScaleTransform,
    QuantileTransform,
)
from dcf.trainer import CoordBestValTrainer, BestValTrainer
from .common import freq_to_context_length, get_input_size, get_dim_time_feat


def nnar(
    freq,
    prediction_length,
    target_dim,
    opt_args,
    ds_name,
    out_path,
    len_train,
    len_val,
    len_test,
    ds_train_mv,
    stationary,
    seed,
    opts,
    train_on_val,
    baserun,
    load,
    cont,
    use_val,
    device,
    encoder="LSTM",
    **kwargs
):
    if stationary:
        lr = 1e-3
    else:
        lr = 0.01
    sigma = 1.0
    dim_hidden = 128
    input_size = get_input_size(ds_name)
    dim_time_feat = get_dim_time_feat(freq)
    if "dequan" in opts:
        dequantize = [-0.5, 0.5]
    else:
        dequantize = None
    if "constantscale" in opts:
        constant_scale = True
    else:
        constant_scale = False
    if stationary:
        patience = 20
    else:
        patience = 20
    eval_batch_size = 8
    n_sample_dim = 1
    n_sample_time = None
    n_batches = 10
    epochs = 100
    scaling = False
    quantile = False
    len_context = prediction_length
    dropout_rate = 0.0
    if ds_name == "exchange_rate_nips":
        # daily
        len_context = freq_to_context_length(freq, prediction_length)
        dim_hidden = 8
        batch_size = 128
        n_sample_dim = 8
        scaling = True
        if stationary:
            epochs = 50
        else:
            epochs = 60
            sigma = 100
        constant_scale = True
    elif ds_name == "solar_nips":
        # hourly
        len_context = freq_to_context_length(freq, prediction_length)
        batch_size = 32
        n_sample_dim = 10
        if stationary:
            if constant_scale:
                epochs = 240
                lr = 0.01 
            else:
                epochs = 40
                lr = 0.001
        else:
            if constant_scale:
                epochs = 40
                sigma = 100
            else:
                epochs = 140
                sigma = 100
    elif ds_name == "electricity_nips":
        # hourly
        len_context = freq_to_context_length(freq, prediction_length)
        batch_size = 32
        n_sample_dim = 10
        if stationary:
            if constant_scale:
                epochs = 130
            else:
                epochs = 110
        else:
            epochs = 180
            sigma = 100
    elif ds_name == "traffic_nips":
        # hourly
        len_context = freq_to_context_length(freq, prediction_length)
        batch_size = 32
        eval_batch_size = 1
        n_sample_dim = 20
        n_sample_time = 2000
        quantile = True
        if stationary:
            if constant_scale:
                epochs = 200
                lr = 0.01
            else:
                epochs = 350
                lr = 0.01
        else:
            epochs = 60
            sigma = 0.01
    elif ds_name == "taxi_30min":
        # half hourly
        len_context = prediction_length
        batch_size = 16
        eval_batch_size = 16
        n_sample_dim = 10
        if stationary:
            if constant_scale:
                epochs = 60
            else:
                epochs = 50
        else:
            epochs = 170
            sigma = 0.01
    elif ds_name == "wiki-rolling_nips":
        # daily
        len_context = freq_to_context_length(freq, prediction_length)
        eval_batch_size = 1
        n_sample_dim = 20
        quantile = True
        if stationary:
            batch_size = 16
            if constant_scale:
                epochs = 20
                lr = 0.01
            else:
                epochs = 10
                lr = 0.01
        else:
            batch_size = 4
            epochs = 20
            sigma = 100
    elif ds_name == "walmart_norm":
        len_context = freq_to_context_length(freq, prediction_length)
        batch_size = 32
        dim_hidden = 2048
        n_sample_dim = 20
        constant_scale = True
        if stationary:
            epochs = 60
        else:
            epochs = 80
            sigma = 100
    elif ds_name == "temperature_norm":
        len_context = freq_to_context_length(freq, prediction_length)
        batch_size = 32
        dim_hidden = 512
        n_sample_dim = 20
        constant_scale = True
        if stationary:
            epochs = 30
            lr = 0.01
        else:
            epochs = 200
            sigma = 100
    else:
        raise NotImplementedError
    if stationary:
        opt_args["val_rolling"] = prediction_length
    else:
        opt_args["val_rolling"] = prediction_length
    opt_args["multi_dim"] = target_dim
    if "plot_latent" not in kwargs:
        opt_args["plot_latent"] = not stationary
    if train_on_val:
        len_total = len_val
    else:
        len_total = len_train
    if kwargs.get("epochs"):
        epochs = kwargs["epochs"]
    if kwargs.get("lr"):
        lr = kwargs["lr"]
    if kwargs.get("sigma"):
        sigma = kwargs["sigma"]
    dim_hidden = kwargs.get("dim_hidden") or dim_hidden
    print(f"lr = {lr}, epochs = {epochs}")
    dist = "Normal"
    if use_val:
        epochs = kwargs["epochs"]
    if "scale" in opts:
        scaling = True
    elif "noscale" in opts:
        scaling = False
    if "quantile" in opts:
        quantile = True
    elif "noquantile" in opts:
        quantile = False
    if "ctx" in opts:
        len_context = freq_to_context_length(freq, prediction_length)
    elif "noctx" in opts:
        len_context = prediction_length
    hyperparams = {
        "freq" :freq,
        "len_context": len_context,
        "len_pred": prediction_length,
        "dim_target": target_dim,
        "dim_input": input_size,
        "dim_hidden": dim_hidden,
        "dim_time_feat": dim_time_feat,
        "use_identity": False,
        "opts": opts,
        "dist": dist,
        "encoder": encoder,
        "dropout_rate": dropout_rate,
        "eval_batch_size": eval_batch_size,
    }
    ds_train_mv = list(ds_train_mv)
    assert len(ds_train_mv) == 1
    target = ds_train_mv[0]["target"]
    if quantile:
        quantile_ds = {
            "exchange_rate_nips": [0.025, 0.975],
            "solar_nips": [0.025, 0.975],
            "electricity_nips": [0.025, 0.975],
            "traffic_nips": [0.025, 0.975],
            "taxi_30min": [0.025, 0.975],
            "wiki-rolling_nips": [0.025, 0.975],
        }
        hyperparams["scaling"] = QuantileTransform(quantile_ds[ds_name])
    elif scaling:
        rolling_std = pd.DataFrame(target.T).rolling(len_context).std(ddof=0)
        global_std = rolling_std.replace(0, np.NaN).mean(0)
        default_std = global_std.mean(0)
        global_std = np.nan_to_num(global_std.values, nan=default_std)
        hyperparams["scaling"] = ShiftScaleTransform(global_std)
    if not scaling:
        m = target.mean(1)
        s = target.std(1)
        hyperparams["invnet"] = ShiftScaleLayer(m=m, s=s)
    hyperparams["dequantize"] = dequantize
    if stationary:
        return NAREstimator(
            trainer=BestValTrainer(
                epochs=epochs,
                num_batches_per_epoch=n_batches,
                batch_size=batch_size,
                learning_rate=lr,
                device=device,
                savepath=out_path,
                load=load,
                patience=patience,
            ),
            len_infer_model=prediction_length,
            constant_scale=constant_scale,
            dim_dynamic_in=4,
            n_sample=100,
            **hyperparams,
        )
    else:
        if baserun:
            loadpath = Path(baserun)/ds_name/str(seed)/"model"
        else:
            loadpath = None
        print(f"loadpath: {loadpath}")
        print(f"sigma = {sigma}")
        len_infer_model = prediction_length
        num_batches_per_epoch = 10
        n_update = 10
        if "debug" in opts:
            num_batches_per_epoch = 2
            n_update = 2
        trainer = CoordBestValTrainer(
            epochs=epochs,
            num_batches_per_epoch=num_batches_per_epoch,
            batch_size=batch_size,
            learning_rate=lr,
            device=device,
            savepath=out_path,
            load=load,
            n_update=n_update,
            patience=patience,
            cont=cont,
            learning_rate_model=1e-3,
        )
        len_lookback = min(len_test) - prediction_length
        len_lookback = min((len_lookback, len_train - 5 * prediction_length))
        return NNAREstimator(
            load_path=loadpath,
            trainer=trainer,
            len_infer_post=None,
            len_infer_model=len_infer_model,
            dim_dynamic_in=4,
            n_sample_param=20,
            n_sample_target=100,
            n_sample_dim=n_sample_dim,
            n_sample_time=n_sample_time,
            step_param=1,
            len_total=len_total,
            len_lookback=len_lookback,
            constant_scale=constant_scale,
            kwargs_prior={
                "sigma_ub": sigma,
                "sigma_d_ub": sigma * 0.1,
            },
            **hyperparams,
        )

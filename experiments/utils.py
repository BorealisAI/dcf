# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import math
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #ignore pandas future warnings
import logging
logging.basicConfig(level=logging.INFO)

import shutil
import argparse
import json
import random
from datetime import datetime

from gluonts.evaluation import make_evaluation_predictions, Evaluator, MultivariateEvaluator
from gluonts.dataset.common import ListDataset, load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mxnet as mx
import torch
from pts.model import PyTorchEstimator
from gluonts.mx import GluonEstimator

from .constants import EXP_FOLDER
from . import preprocess as preproc
from . import plotting


def create_common_parser(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--ds", type=str, required=True, help="Dataset name")
    parser.add_argument("--model", type=str, default="NNAR", help="Model name")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=None, help="Maximum number of epochs")
    parser.add_argument("--nbatches", type=int, default=50, help="Number of batches per epoch")
    parser.add_argument("--sample", type=int, default=100, help="Number of samples to draw per instance")
    parser.add_argument("--run", type=str, default=None, help="Folder name to store the output of the run")
    parser.add_argument("--baserun", type=str, default=None, help="Base run name to load NAR from")
    parser.add_argument("--cpu", action="store_true", help="Run on CPU")
    parser.add_argument("--force", action="store_true", help="Force rerun (overwrite previous model)")
    parser.add_argument("--plot", action="store_true", help="Plot latent variable trajectory only")
    parser.add_argument("--eval", action="store_true", help="Eval only")
    parser.add_argument("--load", type=int, default=0, help="Load best(1)/last(1) model")
    parser.add_argument("--cont", action="store_true", help="Should continue training after loading")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--sigma", type=float, default=None, help="Prior scale")
    parser.add_argument("--dim-hidden", type=int, default=None, help="Model size")
    return parser


def set_seed(seed):
    if seed is not None:
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            import mxnet as mx
            mx.random.seed(seed)
        except Exception as e:
            print("mxnet import error", e)


def get_exp_folder(exp_name, model_name, seed, run_name=None, base_path=EXP_FOLDER, prefix_path=None):
    if run_name is None:
        run_name = f"run_{datetime.now().isoformat()}"
    if prefix_path:
        base_path = base_path / prefix_path
    folder = base_path / model_name / run_name / exp_name / str(seed)
    return folder


def run_synthetic_experiment(
    seed,
    exp_name,
    dataset,
    model_name,
    estimator_init,
    len_pred,
    len_test=1000,
    len_val=500,
    freq="s",
    start="2020-01-01",
    run_name=None,
    train_on_val=False,
    **kwargs,
):
    set_seed(seed)
    ds_train, ds_val, ds_test = preproc.split_synthetic_dataset(dataset, start=start, freq=freq, len_test=len_test, len_val=len_val)
    out_path = get_exp_folder(exp_name, model_name, seed, run_name, prefix_path="synthetic")
    args = kwargs.copy()
    dim_target = len(ds_train)
    grouper = MultivariateGrouper(num_test_dates=1, max_target_dim=dim_target)
    ds_train_mv = grouper(ds_train)
    len_train = next(iter(ds_train_mv))[FieldName.TARGET].shape[1]
    grouper = MultivariateGrouper(num_test_dates=1, max_target_dim=dim_target)
    ds_val_mv = grouper(ds_val)
    len_val = next(iter(ds_val_mv))[FieldName.TARGET].shape[1]
    grouper = MultivariateGrouper(num_test_dates=len(ds_test)/len(ds_train), max_target_dim=dim_target)
    ds_test_mv = grouper(ds_test)
    args.update(
        len_train=len_train,
        len_val=len_val,
        out_path=out_path,
        len_pred=len_pred,
        val_rolling=len_pred,
        train_on_val=train_on_val,
        dim_target=dim_target,
        ds_train_mv=ds_train_mv,
        seed=seed,
    )
    print(f"training length: {len_train}")
    print(f"validation length: {len_val}")
    estimator = estimator_init(args)
    if args.get("multi_dim"):
        ds_train = ds_train_mv
        ds_val = ds_val_mv
        ds_test = ds_test_mv
    return run_experiment(
        estimator=estimator,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        ds_params=dataset.params,
        **args,
    )


def run_benchmark_experiment(
    seed,
    ds_name,
    model_name,
    estimator_init,
    n_val_window=None,
    run_name=None,
    **kwargs
):
    set_seed(seed)
    exp_name = ds_name
    try:
        ds = get_dataset(ds_name)
    except:
        dataset_path = get_download_path() / "datasets" / ds_name
        ds = load_datasets(
            metadata=dataset_path,
            train=dataset_path / "train",
            test=dataset_path / "test",
        )
    ds_train = ds.train
    ds_test = ds.test
    prediction_length = ds.metadata.prediction_length
    instance = next(iter(ds_train))
    freq = instance[FieldName.START].freqstr
    print(f"freq: {freq}")
    target = instance[FieldName.TARGET]
    assert len(target.shape) == 1
    if ds_name == "electricity_nips":
        # hourly
        assert prediction_length == 24
    elif ds_name == "exchange_rate_nips":
        # daily
        assert prediction_length == 30
    elif ds_name == "solar_nips":
        # hourly
        assert prediction_length == 24
    elif ds_name == "taxi_30min":
        # half hourly
        assert prediction_length == 24
    elif ds_name == "traffic_nips":
        # hourly
        assert prediction_length == 24
    elif ds_name == "wiki-rolling_nips":
        # daily
        assert prediction_length == 30
    elif ds_name.startswith("walmart"):
        assert prediction_length == 4
    elif ds_name.startswith("temperature"):
        assert prediction_length == 3
    else:
        raise NotImplementedError
    # notice that train/val and test may not be contiguous (e.g., taxi)
    target_dim = min(2000, int(ds.metadata.feat_static_cat[0].cardinality))
    print(f"Cardinality: {target_dim}")
    test_grouper = MultivariateGrouper(
        num_test_dates=int(len(ds.test)/len(ds.train)),
        max_target_dim=target_dim
    )
    ds_test_mv = test_grouper(ds_test)
    n_test_window = len(ds_test_mv)
    print(f"Total number of prediction windows in test set: {n_test_window}")
    len_test = [x[FieldName.TARGET].shape[1] for x in ds_test_mv]
    for i, l in enumerate(len_test):
        print(f"Test {i+1} length: {l}")
    grouper = MultivariateGrouper(num_test_dates=1, max_target_dim=target_dim)
    ds_train_mv = grouper(ds_train)
    instance = next(iter(ds_train_mv))
    start_date = instance[FieldName.START] 
    len_train = instance[FieldName.TARGET].shape[1]
    # note that split_date is inclusive for the training split
    n_train_window = len_train // prediction_length
    print(f"Total number of prediction windows in training set: {n_train_window}")
    if n_val_window is None:
        n_val_window = math.floor(n_train_window * 0.1)
    print(f"Total number of prediction windows in validation set: {n_val_window}")
    split_date = start_date + (len_train - 1 - prediction_length * n_val_window) * start_date.freq
    split = preproc.SimpleSplitter(
        prediction_length=prediction_length,
        split_date=split_date,
    ).split(ds_train)
    ds_train = ListDataset(split.train, freq=freq)
    ds_val = ListDataset(split.test, freq=freq)
    grouper = MultivariateGrouper(num_test_dates=1, max_target_dim=target_dim)
    ds_train_mv = grouper(ds_train)
    grouper = MultivariateGrouper(num_test_dates=1, max_target_dim=target_dim)
    ds_val_mv = grouper(ds_val)
    len_train = [x[FieldName.TARGET].shape[1] for x in ds_train_mv]
    len_val = [x[FieldName.TARGET].shape[1] for x in ds_val_mv]
    assert len(len_train) == 1
    assert len(len_val) == 1
    len_train = len_train[0]
    len_val = len_val[0]
    print(f"Training length: {len_train}")
    print(f"Validation length: {len_val}")
    opt_args = {
        "val_rolling": prediction_length,
    }
    out_path = get_exp_folder(exp_name, model_name, seed, run_name, prefix_path="benchmark")
    estimator = estimator_init(
        freq=freq,
        prediction_length=prediction_length,
        feat_static_cat=ds.metadata.feat_static_cat,
        len_train=len_train,
        len_val=len_val,
        len_test=len_test,
        ds_name=ds_name,
        out_path=out_path,
        target_dim=target_dim,
        opt_args=opt_args,
        ds_train_mv=ds_train_mv,
        seed=seed,
        **kwargs,
    )
    kwargs.update(opt_args)
    if opt_args.get("multi_dim"):
        ds_train = ds_train_mv
        ds_val = ds_val_mv
        ds_test = ds_test_mv
    return run_experiment(
        seed=seed,
        out_path=out_path,
        estimator=estimator,
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        len_pred=prediction_length,
        len_train=len_train,
        len_val=len_val,
        **kwargs,
    )


def save_model(model, path):
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path)
    elif isinstance(model, mx.gluon.Block):
        model.save_parameters(str(path))
    else:
        raise NotImplementedError(f"Saving model of type {type(model)} is not supported.")
    print(f"Model saved to {path}")


def load_model(estimator, path):
    if isinstance(estimator, PyTorchEstimator):
        device = estimator.trainer.device
        model = estimator.create_training_network(device)
        model.load_state_dict(torch.load(path))
        transformation = estimator.create_transformation()
        predictor = estimator.create_predictor(transformation, model, device)
    elif isinstance(estimator, GluonEstimator):
        model = estimator.create_training_network()
        model.load_parameters(str(path))
        transformation = estimator.create_transformation()
        with estimator.trainer.ctx:
            predictor = estimator.create_predictor(transformation, model)
    else:
        raise NotImplementedError(f"Loading model of from estimator of type {type(estimator)} is not supported.")
    return predictor, model


def run_experiment(
    seed,
    out_path,
    estimator,
    ds_train,
    ds_val,
    ds_test,
    len_pred,
    len_context=None,
    n_sample_test=100,
    multi_dim=None,
    force=False,
    train_rolling=None,
    val_rolling=None,
    test_rolling=None,
    len_train=None,
    len_val=None,
    n_fig_max=500,
    train_on_val=False,
    use_val=True,
    skip_eval=False,
    test_on_val=False,
    test_on_train=False,
    plot_latent=False,
    plot_latent_with_datetime=False,
    ds_params=None,
    cont=False,
    **kwargs,
):
    print(f"seed = {seed}")
    set_seed(seed)
    is_multi_dim = multi_dim is not None

    if train_on_val:
        print("Train on training+validation set.")
        ds_train = ds_val
        if use_val:
            ds_val = ds_test
            ds_val_test = ds_test
            val_rolling = False
    if test_rolling:
        ds_test = preproc.generate_rolling(ds_test, len_pred, len_val, multi_dim=is_multi_dim)
    if train_rolling:
        ds_train = preproc.generate_rolling(ds_train, train_rolling, 0, multi_dim=is_multi_dim)
    ds_val_org = ds_val
    if use_val and val_rolling:
        if train_on_val:
            ds_val = ds_test
        else:
            ds_val = preproc.generate_rolling(ds_val_org, val_rolling, len_train, multi_dim=is_multi_dim)
    if not train_on_val and test_on_val:
        ds_val_test = preproc.generate_rolling(ds_val_org, len_pred, len_train, multi_dim=is_multi_dim)
    if test_on_train:
        # should really skip first k = len_contex + max(lag_seq) steps
        # but it'd be estimator dependent
        # so we skip first half as a heuristic
        ds_train_test = preproc.generate_rolling(ds_train, len_pred, len_train - 5 * len_pred, multi_dim=is_multi_dim)

    quantiles = (np.arange(20)/20.0)[1:]
    if multi_dim is not None and multi_dim > 1:
        evaluator = MultivariateEvaluator(quantiles=quantiles, target_agg_funcs={'sum': np.sum})
    else:
        evaluator = Evaluator(quantiles=quantiles)
    if force and out_path.exists():
        print(f"Removing output path: {out_path}")
        shutil.rmtree(out_path)

    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "model"
    if not cont and model_path.is_file():
        predictor, model = load_model(estimator, model_path)
    else:
        if not use_val:
            output = estimator.train_model(ds_train)
        else:
            print(len(list(ds_train)))
            print(list(ds_train)[0]["target"].shape)
            output = estimator.train_model(ds_train, ds_val)
        predictor = output.predictor
        if output.trained_net:
            save_model(output.trained_net, out_path / "model")
        model = output.trained_net

    if len_context is None:
        if hasattr(estimator, "context_length"):
            len_context = estimator.context_length
        elif hasattr(estimator, "len_context"):
            len_context = estimator.len_context
        else:
            len_context = len_pred

    def eval(ds_test, out_path):
        out_path.mkdir(parents=True, exist_ok=True)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=ds_test,
            predictor=predictor,
            num_samples=n_sample_test,
        )
        forecasts = list(forecast_it)
        tss = list(ts_it)
        agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts), num_series=len(ds_test))
        with open(out_path / "agg_metrics.json", "w") as f:
            json.dump(agg_metrics, f, indent=4)
        item_metrics.to_csv(out_path / "item_metrics.csv")
        if multi_dim and multi_dim == 1 and len(forecasts[0].samples.shape) > 2:
            forecasts = list(f.copy_dim(0) for f in forecasts)
        if multi_dim and multi_dim > 1:
            for i in range(min(n_fig_max, len(tss))):
                ts = tss[i]
                forecast = forecasts[i]
                seq_len, target_dim = ts.shape
                # for dim in [79, 261, 303]:
                for dim in range(min(20, target_dim)):
                    plotting.plot_prob_forecasts(ts.iloc[:, dim], forecast.copy_dim(dim), plot_length=len_context+len_pred)
                    plt.savefig(out_path / f"forecast_{dim}_{i}.png")
                    plt.close()
        else:
            for i in range(min(n_fig_max, len(tss))):
                ts = tss[i]
                forecast = forecasts[i]
                plotting.plot_prob_forecasts(ts, forecast, plot_length=len_context+len_pred)
                plt.savefig(out_path / f"forecast_{i}.png")
                plt.close()
    
    if not skip_eval:
        if not train_on_val and test_on_val:
            set_seed(seed)
            eval(ds_val_test, out_path / "val")
        set_seed(seed)
        eval(ds_test, out_path / "test")
        if test_on_train:
            set_seed(seed)
            eval(ds_train_test, out_path / "train")
    
    if plot_latent:
        plotting.plot_latent(
            estimator=estimator,
            net=predictor.prediction_net,
            dss={
                "train": ds_train,
                "test": ds_test,
            },
            base_fig_path=out_path / "latent",
            ds_params=ds_params,
            use_datetime=plot_latent_with_datetime,
        )

    return estimator, predictor, model

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import json
import argparse
from collections import OrderedDict
import re

import pandas as pd
import numpy as np

from experiments.constants import EXP_FOLDER


def average_crps(result):
    total = 0.0
    n_total = 0
    for k, v in result.items():
        if re.fullmatch("[0-9]+_mean_wQuantileLoss", k):
            n_total += 1
            total += v
    return total / n_total
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate result tables")
    parser.add_argument("--prefix", type=str, default="benchmark", help="Synthetic or benchmark")
    parser.add_argument("--run", type=str, nargs="+", default=None, help="Run name")
    parser.add_argument("--model", type=str, nargs="+", default=None, help="Optional list of models to include")
    parser.add_argument("--modelrun", type=str, nargs="+", default=None, help="Alternative specification of model:run combinations")
    parser.add_argument("--ds", type=str, nargs="+", default=None, help="Optional list of datasets to include")
    parser.add_argument("--split", type=str, nargs="+", default=["test"], help="Optional list of splits to include")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)), help="Optional list of seeds to include")
    args = parser.parse_args()
    prefix = args.prefix
    model_runs = OrderedDict() 
    if args.modelrun:
        for mr in args.modelrun:
            model, runs = mr.split(":")
            runs = runs.split(',')
            model_runs[model] = runs
    else:
        runs = args.run
        models = args.model
        for model in models:
            model_runs[model] = runs
    models = model_runs.keys()
    datasets = args.ds
    seeds = args.seeds
    base_path = EXP_FOLDER / prefix
    metric_file = "agg_metrics.json"
    splits = args.split
    metrics = ["MSE", "mean_wQuantileLoss", "m_sum_mean_wQuantileLoss"]
    full_results = []
    for model_path in base_path.iterdir():
        model_name = model_path.name
        if model_name not in model_runs:
            continue
        for run_path in model_path.iterdir():
            run = run_path.name
            if not run_path.is_dir():
                continue
            if model_runs[model_name] and not any(run.startswith(name) for name in model_runs[model_name]):
                continue
            for ds_path in run_path.iterdir():
                if not ds_path.is_dir():
                    continue
                ds_name = ds_path.name
                if datasets and ds_name not in datasets:
                    continue
                for seed in seeds:
                    tune_path = ds_path / str(seed) / "epoch"
                    if not tune_path.is_dir():
                        continue
                    for epoch_path in tune_path.iterdir(): 
                        epoch = epoch_path.name
                        fpath = epoch_path / metric_file
                        if not fpath.is_file():
                            continue
                        with open(fpath) as f:
                            js = json.load(f)
                            result = {k: js.get(k, np.NaN) for k in metrics}
                            result["dataset"] = ds_name
                            result["model"] = model_name
                            result["run"] = run
                            result["epoch"] = int(epoch)
                            result["seed"] = seed
                            full_results.append(result)
    df = pd.DataFrame.from_dict(full_results, "columns")
    df = df.set_index(["dataset", "model", "run", "epoch", "seed"]).sort_index()
    with pd.option_context(
        "display.max_rows", 500,
        "display.float_format", "{:.4g}".format,
    ):
        print(df)
    df.to_csv(base_path / f"metrics.csv")
    df.to_latex(base_path / f"metrics.tex")
    df = df.reindex(labels=models, axis=0, level=1)
    for metric in metrics:
        d = df[metric]
        d = d.unstack(level="dataset")
        d = d.groupby(level=("model", "run", "epoch")).agg(["mean", "std", "count"])
        if datasets:
            d = d[datasets]
        float_format = {
            "MSE": lambda x: "{:.3e}".format(x).replace("e-0", "e-").replace("e+0","e"),
            "mean_wQuantileLoss": "{:.3f}".format,
        }[metric]
        with pd.option_context("display.float_format", float_format):
            d.to_csv(base_path / f"{metric}.csv")
        d = d.stack(level=0)
        d["mean"] = d["mean"].apply(float_format)
        d["std"] = d["std"].apply(float_format)
        d["mean_std"] = d["mean"] + '$' + '\sd{' + d["std"] + '}$'
        d = d["mean_std"].unstack(level=-1)
        if datasets:
            idx = [ds for ds in datasets if ds in d.columns]
            d = d[idx]
        d.index = d.index.map("{0[0]}_{0[1]}".format)
        d.index = d.index.str.replace("_", "-")
        d.columns = d.columns.str.replace("_", "-")
        d.to_latex(base_path / f"{metric}.tex", escape=False)
# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
import math

from gluonts.model.deepvar import DeepVAREstimator
from gluonts.model.gpvar import GPVAREstimator
from gluonts.mx import Trainer
import pts
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts.model.tempflow import TempFlowEstimator
from pts.model.time_grad import TimeGradEstimator

from dcf.trainer import BestValTrainer
from dcf.mx.model import ValTransformerTempFlowEstimator, ValTempFlowEstimator, ValTimeGradEstimator
from .common import freq_to_context_length, get_input_size


def deep_var(freq, prediction_length, target_dim, opt_args, ctx, ds_name, **kwargs):
    opt_args["multi_dim"] = target_dim
    n_batches = 10
    epochs = 400
    lr = 1e-3
    scaling = False
    num_cells = 32
    len_context = freq_to_context_length(freq, prediction_length)
    if ds_name == "walmart_norm":
        scaling = True
        num_cells = 128
        lr = 1e-3
    elif ds_name == "temperature_norm":
        scaling = True
        num_cells = 128
        lr = 1e-3
    else:
        raise NotImplementedError
    epochs = kwargs.get("epochs") or epochs
    lr = kwargs.get("lr") or lr
    num_cells = kwargs.get("dim_hidden") or num_cells
    mx_trainer = Trainer(ctx, epochs, num_batches_per_epoch=n_batches, learning_rate=lr)
    return DeepVAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=len_context,
        trainer=mx_trainer,
        target_dim=target_dim,
        scaling=scaling,
        num_cells=num_cells,
    )


def gpvar(freq, prediction_length, target_dim, opt_args, ctx, ds_name, **kwargs):
    opt_args["multi_dim"] = target_dim
    n_batches = 10
    epochs = 400
    lr = 1e-3
    scaling = False
    num_cells = 32
    len_context = freq_to_context_length(freq, prediction_length)
    target_dim_sample = None
    if ds_name == "walmart_norm":
        num_cells = 128
        lr = 1e-2
    elif ds_name == "temperature_norm":
        num_cells = 128
        lr = 1e-2
        target_dim_sample = 100
    else:
        raise NotImplementedError
    epochs = kwargs.get("epochs") or epochs
    lr = kwargs.get("lr") or lr
    num_cells = kwargs.get("dim_hidden") or num_cells
    mx_trainer = Trainer(ctx, epochs, num_batches_per_epoch=n_batches, learning_rate=lr)
    return GPVAREstimator(
        freq=freq,
        prediction_length=prediction_length,
        context_length=len_context,
        trainer=mx_trainer,
        target_dim=target_dim,
        scaling=scaling,
        num_cells=num_cells,
        target_dim_sample=target_dim_sample,
    )


def lstm_maf(
    freq,
    prediction_length,
    target_dim,
    opt_args,
    out_path,
    len_train,
    len_val,
    ds_name,
    cell_type,
    opts,
    device,
    **kwargs
):
    opt_args["multi_dim"] = target_dim
    # following paper
    epochs = 40
    n_batches = 100
    lr = 1e-3
    n_blocks = 3
    batch_size = 64
    scaling = True
    dequantize = False
    print(f"cell type: {cell_type}")
    len_context = freq_to_context_length(freq, prediction_length)
    input_size = get_input_size(ds_name)
    n_cell = 40
    len_cond = 200
    if ds_name == "walmart_norm":
        scaling = False
        n_cell = 2048
        batch_size = 32
        n_batches = 10
        lr = 1e-2
        epochs = 80
    elif ds_name == "temperature_norm":
        scaling = False
        n_cell = 512
        batch_size = 32
        n_batches = 10
        lr = 1e-3
        epochs = 280
    else:
        raise NotImplementedError
    epochs = kwargs.get("epochs") or epochs
    lr = kwargs.get("lr") or lr
    n_cell = kwargs.get("dim_hidden") or n_cell
    if "val" in opts:
        len_val = len_val - len_train
        n_batches_val = math.ceil((len_val//prediction_length) / batch_size)
        print(f"len_val = {len_val}, len_pred = {prediction_length}, batch_size = {batch_size}, n_batches_val = {n_batches_val}")
        trainer = BestValTrainer(
            device=device,
            epochs=epochs,
            learning_rate=lr,
            num_batches_per_epoch=n_batches,
            n_batches_val=n_batches_val,
            batch_size=batch_size,
            savepath=out_path,
        )
        cls = ValTempFlowEstimator 
    else:
        trainer=pts.Trainer(
            device=device,
            epochs=epochs,
            learning_rate=lr,
            num_batches_per_epoch=n_batches,
            batch_size=batch_size,
        )
        cls = TempFlowEstimator
    return cls(
        cell_type=cell_type,
        num_cells=n_cell,
        n_blocks=n_blocks,
        input_size=input_size,
        target_dim=target_dim,
        context_length=len_context,
        prediction_length=prediction_length,
        conditioning_length=len_cond,
        scaling=scaling,
        flow_type="MAF",
        dequantize=dequantize,
        freq=freq,
        trainer=trainer,
    )


def transformer_maf(
    freq,
    prediction_length,
    target_dim,
    opt_args,
    out_path,
    len_train,
    len_val,
    ds_name,
    opts,
    device,
    **kwargs
):
    opt_args["multi_dim"] = target_dim
    # following paper
    epochs = 40
    n_batches = 100
    lr = 1e-3
    n_heads = 8
    n_blocks = 3
    batch_size = 64
    scaling = True
    dequantize = False
    len_context = freq_to_context_length(freq, prediction_length)
    input_size = get_input_size(ds_name)
    if ds_name == "walmart_norm":
        scaling = False
        d_model = 128
        batch_size = 32
        n_batches = 10
        lr = 1e-3
        epochs = 50
    elif ds_name == "temperature_norm":
        scaling = False
        d_model = 32
        batch_size = 32
        n_batches = 10
        lr = 1e-3
        epochs = 180
    else:
        raise NotImplementedError
    epochs = kwargs.get("epochs") or epochs
    lr = kwargs.get("lr") or lr
    d_model = kwargs.get("dim_hidden") or d_model
    if "val" in opts:
        len_val = len_val - len_train
        n_batches_val = math.ceil((len_val//prediction_length) / batch_size)
        print(f"len_val = {len_val}, len_pred = {prediction_length}, batch_size = {batch_size}, n_batches_val = {n_batches_val}")
        trainer = BestValTrainer(
            device=device,
            epochs=epochs,
            learning_rate=lr,
            num_batches_per_epoch=n_batches,
            n_batches_val=n_batches_val,
            batch_size=batch_size,
            savepath=out_path,
        )
        cls = ValTransformerTempFlowEstimator 
    else:
        trainer=pts.Trainer(
            device=device,
            epochs=epochs,
            learning_rate=lr,
            num_batches_per_epoch=n_batches,
            batch_size=batch_size,
        )
        cls = TransformerTempFlowEstimator 
    return cls(
        n_blocks=n_blocks,
        d_model=d_model,
        num_heads=n_heads,
        input_size=input_size,
        target_dim=target_dim,
        context_length=len_context,
        prediction_length=prediction_length,
        scaling=scaling,
        flow_type="MAF",
        dequantize=dequantize,
        freq=freq,
        trainer=trainer,
    )


def time_grad(
    freq,
    prediction_length,
    target_dim,
    opt_args,
    out_path,
    len_train,
    len_val,
    ds_name,
    opts,
    device,
    **kwargs
):
    opt_args["multi_dim"] = target_dim
    lr = 1e-3
    scaling = False
    n_batches = 10
    batch_size = 32
    len_context = freq_to_context_length(freq, prediction_length)
    input_size = get_input_size(ds_name)
    if ds_name == "walmart_norm":
        n_cell = 128
        epochs = 40
        lr = 1e-3
    elif ds_name == "temperature_norm":
        n_cell = 2048
        epochs = 90
        lr = 1e-3
    else:
        raise NotImplementedError
    epochs = kwargs.get("epochs") or epochs
    lr = kwargs.get("lr") or lr
    n_cell = kwargs.get("dim_hidden") or n_cell
    if "val" in opts:
        len_val = len_val - len_train
        n_batches_val = math.ceil((len_val//prediction_length) / batch_size)
        print(f"len_val = {len_val}, len_pred = {prediction_length}, batch_size = {batch_size}, n_batches_val = {n_batches_val}")
        trainer = BestValTrainer(
            device=device,
            epochs=epochs,
            learning_rate=lr,
            num_batches_per_epoch=n_batches,
            n_batches_val=n_batches_val,
            batch_size=batch_size,
            savepath=out_path,
        )
        cls = ValTimeGradEstimator
    else:
        trainer=pts.Trainer(
            device=device,
            epochs=epochs,
            learning_rate=lr,
            num_batches_per_epoch=n_batches,
            batch_size=batch_size,
        )
        cls = TimeGradEstimator 
    return cls(
        num_cells=n_cell,
        input_size=input_size,
        target_dim=target_dim,
        context_length=len_context,
        prediction_length=prediction_length,
        scaling=scaling,
        freq=freq,
        trainer=trainer,
    )

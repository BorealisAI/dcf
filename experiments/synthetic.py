# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from functools import partial
import math

from gluonts.mx.distribution import GaussianOutput
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.deepstate.issm import CompositeISSM
from gluonts.mx import Trainer
import mxnet as mx

from . import datasets, utils
from .true_model import TrueAREstimator
from dcf.trainer import CoordBestValTrainer, BestValTrainer
from dcf.mx.model import DiagonalMultivariateGaussianOutput, SimplifiedDeepAREstimator, SimplifiedDeepVAREstimator, SimplifiedTransformerTempFlowEstimator
from dcf.model.nnar import NNAREstimator, NAREstimator


if __name__ == "__main__":
    import socket
    print(f"Host: {socket.gethostname()}")
    parser = utils.create_common_parser(desc="Run experiments on a benchmark dataset.")
    args = parser.parse_args()
    epochs = args.epochs
    if epochs is None:
        epochs = 10000
    run_name = args.run
    n_sample = args.sample
    baserun = args.baserun
    opts = run_name.split('_') if run_name else []
    if args.cpu:
        ctx = mx.cpu()
        device = "cpu"
    else:
        ctx = mx.gpu()
        device = "cuda"
    ds_name = args.ds
    exp_name = ds_name
    model_name = args.model
    load = args.load
    freq = "s"
    lags_seq = [1]
    len_pred = 10
    eval_only = args.eval
    if ds_name in ["dynamic_var_1"]:
        dataset = datasets.VARDataSet(ds_name)
        len_context = 200
    else:
        dataset = datasets.ARDataSet(ds_name)
        len_context = 200

    def ground_truth(args):
        dim_target = args["dim_target"]
        if dim_target > 1:
            args["multi_dim"] = dim_target
        return TrueAREstimator(
            freq=freq,
            model=dataset.model,
            device=device,
            len_pred=len_pred,
            len_context=len_context,
            dim_target=dim_target,
        )

    def deep_ar(args, num_cells=40):
        return SimplifiedDeepAREstimator(
            freq=freq,
            prediction_length=len_pred,
            context_length=len_context,
            lags_seq=lags_seq,
            time_features=[],
            scaling=False,
            distr_output=GaussianOutput(),
            num_cells=num_cells,
            trainer=Trainer(
                ctx,
                epochs,
                learning_rate=1e-3,
            )
        )

    def deep_state(args, num_cells=40):
        return DeepStateEstimator(
            freq=freq,
            prediction_length=len_pred,
            past_length=len_context,
            time_features=[],
            cardinality=[],
            scaling=False,
            use_feat_static_cat=False,
            num_cells=num_cells,
            issm=CompositeISSM([], False),
            trainer=Trainer(
                ctx,
                epochs,
                learning_rate=1e-3,
            )
        )

    # multivariate models
    def deep_var(args, num_cells=40):
        dim_target = args["dim_target"]
        args["multi_dim"] = dim_target
        mx_trainer = Trainer(ctx, epochs, learning_rate=1e-3)
        return SimplifiedDeepVAREstimator(
            freq=freq,
            prediction_length=len_pred,
            context_length=len_context,
            lags_seq=lags_seq,
            time_features=[],
            scaling=False,
            trainer=mx_trainer,
            target_dim=dim_target,
            num_cells=num_cells,
            distr_output=DiagonalMultivariateGaussianOutput(dim_target),
        )

    def transformer_maf(args, d_model=32):
        dim_target = args["dim_target"]
        args["multi_dim"] = dim_target
        lr = 1e-3
        n_heads = 4
        batch_size = 32
        input_size = 9
        assert "val" in opts
        len_val = args["len_val"] - args["len_train"]
        n_batches_val = math.ceil((len_val//len_pred) / batch_size)
        trainer = BestValTrainer(
            device=device,
            epochs=epochs,
            learning_rate=lr,
            num_batches_per_epoch=50,
            n_batches_val=n_batches_val,
            batch_size=batch_size,
            early_stopping=True,
            val_freq=1,
            savepath=args["out_path"],
        )
        return SimplifiedTransformerTempFlowEstimator(
            d_model=d_model,
            num_heads=n_heads,
            input_size=input_size,
            target_dim=dim_target,
            context_length=len_context,
            prediction_length=len_pred,
            lags_seq=lags_seq,
            scaling=False,
            flow_type="MAF",
            dequantize=False,
            time_features=[],
            freq=freq,
            trainer=trainer,
        )

    def nnar(args, stationary=False, encoder=None, dim_hidden=32, use_identity=True):
        dim_target = args["dim_target"]
        args["multi_dim"] = dim_target
        args["plot_latent"] = not stationary and not eval_only
        batch_size = 32
        len_context = 1
        len_hist = len_context + max(lags_seq)
        hyperparams = {
            "freq" :freq,
            "len_context": len_context,
            "len_pred": len_pred,
            "dim_target": dim_target,
            "dim_input": dim_target,
            "dim_dynamic_in": 4,
            "dim_hidden": dim_hidden,
            "use_identity": use_identity,
            "opts": opts,
            "scaling": None,
            "encoder": encoder,
            "eval_batch_size": batch_size,
            "lags_seq": lags_seq,
        }
        if stationary:
            len_infer_model = len_pred
            return NAREstimator(
                trainer=BestValTrainer(
                    epochs=epochs,
                    num_batches_per_epoch=50,
                    batch_size=batch_size,
                    learning_rate=1e-3,
                    early_stopping=True,
                    val_freq=1,
                    device=device,
                    savepath=args["out_path"],
                    load=load,
                    multi_dim=(dim_target > 1),
                ),
                len_infer_model=len_infer_model,
                n_sample=n_sample,
                time_features=[],
                constant_scale=True,
                **hyperparams,
            )
        seed = args["seed"]
        if baserun is not None:
            if encoder == "MLP":
                basemodel = "MLPNAR"
            else:
                basemodel = "NAR"
            loadpath = args["out_path"].parents[2]/basemodel/baserun/str(seed)/"model"
        else:
            loadpath = None
        if args["train_on_val"]:
            len_total = args["len_val"]
        else:
            len_total = args["len_train"]
        len_infer = len_total - len_hist
        cls = NNAREstimator
        trainer = CoordBestValTrainer(
            epochs=epochs,
            num_batches_per_epoch=10,
            batch_size=batch_size,
            learning_rate=1e-2,
            learning_rate_model=1e-3,
            device=device,
            savepath=args["out_path"],
            load=load,
            multi_dim=(dim_target > 1),
            early_stopping=True,
            patience=100,
        )
        len_infer_model = len_pred
        return cls(
            load_path=loadpath,
            trainer=trainer,
            len_infer_model=len_infer_model,
            len_infer_post=len_infer,
            len_total=len_total,
            len_lookback=len_total,
            n_sample_param=n_sample,
            n_sample_target=100,
            time_features=[],
            constant_scale=True,
            kwargs_posterior={
                "use_nn": "nn" in opts,
                "init_a": 0.0,
                "init_s": 1e-1,
                "len_window": 500,
            },
            **hyperparams,
        )

    models = {
        "GroundTruth": ground_truth,
        "DeepAR": deep_ar, # default num_cells=40
        "DeepAR_10": partial(deep_ar, num_cells=10),
        "DeepAR_160": partial(deep_ar, num_cells=160),
        "DeepState": deep_state,
        "DeepVAR": deep_var,
        "DeepVAR_10": partial(deep_var, num_cells=10),
        "DeepVAR_160": partial(deep_var, num_cells=160),
        "TransformerMAF": transformer_maf,
        "TransformerMAF_8": partial(transformer_maf, d_model=8),
        "TransformerMAF_128": partial(transformer_maf, d_model=128),
        "NAR": partial(nnar, stationary=True),
        "NNAR": partial(nnar, stationary=False),
        "MLPNAR": partial(nnar, stationary=True, encoder="MLP"),
        "MLPNNAR": partial(nnar, stationary=False, encoder="MLP"),
    }

    estimator = utils.run_synthetic_experiment(
        seed=args.seed,
        exp_name=exp_name,
        dataset=dataset,
        model_name=model_name,
        estimator_init=models[model_name],
        len_pred=len_pred,
        freq=freq,
        run_name=run_name,
        force=args.force,
        test_rolling=len_pred,
        n_sample_test=1000,
        skip_eval=args.plot,
        use_val=("val" in opts),
        train_on_val=False,
        test_on_train=True,
        len_context=len_pred,
    )

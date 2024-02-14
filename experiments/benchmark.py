# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from functools import partial
import mxnet as mx

from . import utils
from .setup.benchmark.ours import nnar
from .setup.benchmark.baselines import (
    deep_var,
    gpvar,
    transformer_maf,
    lstm_maf,
    time_grad,
)


if __name__ == "__main__":
    import socket
    print(f"Host: {socket.gethostname()}")
    parser = utils.create_common_parser(desc="Run experiments on a benchmark dataset.")
    args = parser.parse_args()
    n_sample = args.sample
    run_name = args.run
    opts = run_name.split('_') if run_name else []
    if args.cpu:
        ctx = mx.cpu()
        device = "cpu"
    else:
        ctx = mx.gpu()
        device = "cuda"

    models = {
        "DeepVAR": deep_var,
        "GPVAR": gpvar,
        "TransformerMAF": transformer_maf,
        "LSTMMAF": partial(lstm_maf, cell_type="LSTM"),
        "GRUMAF": partial(lstm_maf, cell_type="GRU"),
        "TimeGrad": time_grad,
        "NAR": partial(nnar, stationary=True),
        "TransNAR": partial(nnar, stationary=True, encoder="Transformer"),
        "NNAR": partial(nnar, stationary=False),
        "TransNNAR": partial(nnar, stationary=False, encoder="Transformer"),
    }

    utils.run_benchmark_experiment(
        seed=args.seed,
        ds_name=args.ds,
        model_name=args.model,
        estimator_init=models[args.model],
        run_name=args.run,
        force=args.force,
        skip_eval=args.plot,
        use_val=("val" in opts),
        train_on_val=("trainonval" in opts),
        test_on_train=True,
        test_on_val=("val" in opts),
        cont=args.cont,
        load=args.load,
        ctx=ctx,
        device=device,
        lr=args.lr,
        n_batches=args.nbatches,
        epochs=args.epochs,
        baserun = args.baserun,
        opts=opts,
        plot_latent_with_datetime=True,
        sigma=args.sigma,
        dim_hidden=args.dim_hidden,
    )

# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from . import utils


if __name__ == "__main__":
    utils.run_simulation(
        "dynamic_var_1",
        utils.simulate_dynamic_var(len_total=2500, n_per_regime=250, seed=0),
        utils.plot_dynamic_ar
    )
    
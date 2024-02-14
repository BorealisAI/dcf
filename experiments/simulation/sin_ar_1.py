# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from . import utils


if __name__ == "__main__":
    utils.run_simulation(
        "sin_ar_1",
        utils.simulate_sin_ar(len_total=2500, seed=0),
        utils.plot_ts,
    )
    
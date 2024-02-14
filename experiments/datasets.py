# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from pathlib import Path
import pickle

import pandas as pd

from experiments.simulation.utils import ARModel, VARModel

from .constants import DATA_FOLDER


class DataSet:
    BIN_FILE_NAME = "data.pkl"

    def __init__(self, name, path=None):
        self.name = name
        if path is None:
            path = DATA_FOLDER / name
        self.path = Path(path)
        self.df = None
        self.bin_file = self.path / self.BIN_FILE_NAME
    
    def load(self):
        if self.df is None:
            self.df = pd.read_pickle(self.bin_file)
        return self.df


class ARDataSet(DataSet):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        model_path = self.path / "model.pkl"
        if model_path.is_file():
            with open(model_path, "rb") as f:
                params = pickle.load(f)
                self.model = ARModel(
                    params["lag"],
                    params["len_total"],
                    params["coeff"],
                    params["mu"],
                    params["sigma"],
                )
                self.params = params


class VARDataSet(DataSet):
    def __init__(self, name, **kwargs):
        super().__init__(name=name, **kwargs)
        model_path = self.path / "model.pkl"
        if model_path.is_file():
            with open(model_path, "rb") as f:
                params = pickle.load(f)
                self.model = VARModel(
                    params["len_total"],
                    params["coeff"],
                )
                self.params = params

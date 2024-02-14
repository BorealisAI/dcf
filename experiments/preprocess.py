# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from gluonts.dataset.common import ListDataset
from gluonts.dataset.rolling_dataset import truncate_features, StepStrategy
from gluonts.dataset.split import DateSplitter
from gluonts.dataset.field_names import FieldName
import pandas as pd
import numpy as np


def generate_rolling(ds, step_size, start_offset, start=None, multi_dim=True):
    if start is None:
        start = next(iter(ds))[FieldName.START]
    strategy = StepStrategy
    ds_roll = generate_rolling_dataset(
        ds,
        strategy(
            prediction_length=step_size,
            step_size=step_size,
        ),
        start + start_offset * start.freq,
        multi_dim=multi_dim,
    )
    return ds_roll


def split_synthetic_dataset(dataset, len_test, len_val=None, start="2020-01-01", freq="s"):
    df = dataset.load()
    target = df.to_numpy()
    start = pd.Timestamp(start, freq=freq)
    len_total = target.shape[0]
    if len_val is None:
        assert len_test > 10
        len_val = len_test // 10
    len_train = len_total - len_test - len_val
    ds_train = []
    ds_val = []
    ds_test = []
    for j in range(target.shape[1]):
        ds_train.append({
            FieldName.ITEM_ID: j,
            FieldName.START: start,
            FieldName.TARGET: target[:len_train, j],
        })
        ds_val.append({
            FieldName.ITEM_ID: j,
            FieldName.START: start,
            FieldName.TARGET: target[:(len_train+len_val), j],
        })
        ds_test.append({
            FieldName.ITEM_ID: j,
            FieldName.START: start,
            FieldName.TARGET: target[:, j],
        })
    ds_train = ListDataset(ds_train, freq=freq)
    ds_val = ListDataset(ds_val, freq=freq)
    ds_test = ListDataset(ds_test, freq=freq)
    return ds_train, ds_val, ds_test


class SimpleSplitter(DateSplitter):
    def _test_slice(self, item, offset=0):
        return item


def to_df(instance, freq=None):
    target = instance["target"]
    start = instance["start"]
    if not freq:
        freq = start.freqstr
    index = pd.date_range(start=start, periods=target.shape[-1], freq=freq)
    return pd.DataFrame(target.T, index=index)


def generate_rolling_dataset(
    dataset,
    strategy,
    start_time,
    end_time=None,
    multi_dim=True,
):
    ds = []
    for item in dataset:
        df = to_df(item, start_time.freq)
        base = df[:start_time][:-1].to_numpy()
        prediction_window = df[start_time:end_time]

        for window in strategy.get_windows(prediction_window):
            new_item = item.copy()
            new_target = np.concatenate([base, window.to_numpy()])
            if multi_dim:
                new_target = new_target.T
            else:
                new_target = new_target.squeeze(-1)
            new_item[FieldName.TARGET] = new_target

            new_item = truncate_features(
                new_item, len(new_item[FieldName.TARGET])
            )
            ds.append(new_item)

    return ds

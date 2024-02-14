# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from gluonts.model.estimator import Estimator
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.transform import (
    AddAgeFeature,
    Chain,
    TestSplitSampler, 
    InstanceSplitter,
)
from gluonts.dataset.field_names import FieldName
from pts.model.estimator import TrainOutput
from pts.model.utils import get_module_forward_input_names
import numpy as np
import torch
import torch.nn as nn


class TrueARPredictionModel(nn.Module):
    def __init__(self, ar_model, len_pred, len_context, n_sample, dim_target):
        super().__init__()
        self.model = ar_model
        self.len_pred = len_pred
        self.len_context = len_context
        self.n_sample = n_sample
        self.dim_target = dim_target

    def forward(self, past_target, past_feat_dynamic_age, future_feat_dynamic_age):
        # univariate
        # (N, T) -> (N*S, T)
        y = torch.repeat_interleave(past_target, self.n_sample, dim=0)
        # do not modify t0 inplace as it'll modify input
        t0 = future_feat_dynamic_age[:, 0].squeeze()
        for dt in range(self.len_pred):
            t = t0 + dt
            sample = self.model.sample(y, t)[:, None]
            y = torch.cat((y, sample), dim=1)
        sample = y[:, -self.len_pred:]
        if self.dim_target == 1:
            return sample.reshape(
                -1,
                self.n_sample,
                self.len_pred,
            )
        else:
            return sample.reshape(
                -1,
                self.n_sample,
                self.len_pred,
                self.dim_target,
            )


class TrueAREstimator(Estimator):
    def __init__(self, freq, model, device, len_pred, dim_target, len_context=None, n_sample=100, **kwargs):
        super().__init__(**kwargs)
        self.freq = freq
        self.model = TrueARPredictionModel(model, len_pred, len_context, n_sample, dim_target)
        self.device = device
        self.len_pred = len_pred
        if len_context is None:
            len_context = len_pred
        self.len_context = len_context

    def train(self, **kwargs):
        return self.train_model(**kwargs).predictor

    def train_model(self, train_data=None, val_data=None, **kwargs):
        transformation = self.create_transformation()
        prediction_network = self.model.to(self.device)
        prediction_splitter = self.create_instance_splitter()
        input_names = get_module_forward_input_names(prediction_network)
        predictor = PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=1,
            freq=self.freq,
            prediction_length=self.len_pred,
            device=self.device,
        )
        return TrainOutput(
            transformation=transformation,
            trained_net=None,
            predictor=predictor
        )
    
    def create_transformation(self):
        return Chain(
            [
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.len_pred,
                    log_scale=False,
                    dtype=np.int64
                ),
            ]
        )

    def create_instance_splitter(self):
        instance_sampler = TestSplitSampler()

        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=self.len_context,
            future_length=self.len_pred,
            time_series_fields=[
                FieldName.FEAT_AGE,
            ],
        )

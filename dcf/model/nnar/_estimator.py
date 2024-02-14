# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.torch.util import copy_parameters
from gluonts.dataset.field_names import FieldName
from gluonts.time_feature import Constant
from gluonts.transform import (
    SelectFields,
    Transformation,
    Chain,
    InstanceSplitter,
    ValidationSplitSampler,
    TestSplitSampler,
    InstanceSampler,
    RemoveFields,
    AsNumpyArray,
    ExpandDimArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from gluonts.env import env
from gluonts.support.util import maybe_len
from gluonts.transform import Transformation, TransformedDataset
from gluonts.itertools import Cyclic, Cached
from pts.model import PyTorchEstimator
from pts.model.estimator import TrainOutput
from pts.model.utils import get_module_forward_input_names
from pts.feature import (
    fourier_time_features_from_frequency,
    lags_for_fourier_time_features_from_frequency,
)

from ._network import (
    NNARTrainingNetwork,
    NNARPredictionNetwork,
    NARTrainingNetwork,
    NARPredictionNetwork,
)
from dcf.trainer import BestValTrainer


@dataclass
class TrainOutput:
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor
    n_epochs: int = None


class TransformedIterableDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        transform,
        is_train=True,
        cyclic=False,
        cache_data=False,
    ):
        super().__init__()
        if cyclic:
            dataset = Cyclic(dataset)
        if cache_data:
            dataset = Cached(dataset)
        self.transformed_dataset = TransformedDataset(
            dataset,
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        return iter(self.transformed_dataset)


class FixedNumberInstanceSampler(InstanceSampler):
    '''
    Samples a fixed number of points per sequence.
    '''
    n: int = 1

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        assert a <= b
        if a == b:
            i = np.full(self.n, a)
        else:
            i = np.random.randint(a, b, size=self.n)
        return i


class StepWiseInstanceSampler(InstanceSampler):
    '''
    Return a regular time points per sequence.
    '''
    step: int = 1

    def __call__(self, ts: np.ndarray) -> np.ndarray:
        a, b = self._get_bounds(ts)
        if a == b:
            grid = [b]
        else:
            assert a < b
            grid = list(range(a, b, self.step))
            if grid[-1] != b:
                grid.append(b)
        return np.array(grid)


class BaseNAREstimator(PyTorchEstimator):
    def __init__(
        self,
        trainer: BestValTrainer,
        freq,
        len_context,
        len_pred,
        len_infer_model,
        n_sample,
        scaling=None,
        len_pred_scale=None,
        len_context_scale=None,
        use_feat_dynamic_real=False,
        time_features=None,
        lags_seq=None,
        use_identity=False,
        eval_batch_size=32,
        **kwargs,
    ):
        super().__init__(trainer=trainer)
        self.trainer = trainer
        self.len_context = len_context
        self.len_pred = len_pred
        self.len_infer_model = len_infer_model
        self.use_feat_dynamic_real = use_feat_dynamic_real
        self.freq = freq
        self.use_identity = use_identity
        self.n_sample = n_sample
        self.eval_batch_size = eval_batch_size
        self.scaling = scaling
        if len_pred_scale is None:
            len_pred_scale = len_pred
        if scaling:
            self.len_pred_scale = len_pred_scale
            if not len_context_scale:
                len_context_scale = len_context
            self.len_context_scale = len_context_scale
        else:
            self.len_pred_scale = 1
            self.len_context_scale = 1
        self.kwargs = kwargs
        print(f"len_pred: {len_pred}")
        print(f"len_context: {len_context}")
        print(f"len_pred_scale: {len_pred_scale}")
        print(f"len_context_scale: {len_context_scale}")

        self.lags_seq = (
            lags_seq
            if lags_seq is not None
            else lags_for_fourier_time_features_from_frequency(freq_str=freq)
        )
        self.time_features = (
            time_features
            if time_features is not None
            else fourier_time_features_from_frequency(self.freq)
        )
        len_hist = self.len_context + max(self.lags_seq)
        if scaling:
            len_hist = max(len_hist, self.len_context_scale + self.len_pred_scale)
        self.len_hist = len_hist
        if not self.time_features:
            self.time_features = [Constant()]
        self.validation_sampler = ValidationSplitSampler(
            min_past=self.len_hist+self.len_pred,
            min_future=0,
        )
        self.test_sampler = TestSplitSampler(
            min_past=self.len_hist
        )

    def create_transformation(self):
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        
        transforms = [
                RemoveFields(field_names=remove_field_names),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddAgeFeature(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_AGE,
                    pred_length=self.len_pred,
                    log_scale=False,
                    dtype=np.float32
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddTimeFeatures(
                    start_field=FieldName.START,
                    target_field=FieldName.TARGET,
                    output_field=FieldName.FEAT_TIME,
                    time_features=self.time_features,
                    pred_length=self.len_pred,
                ),
                VstackFeatures(
                    output_field=FieldName.FEAT_TIME,
                    input_fields=[FieldName.FEAT_TIME]
                    + (
                        [FieldName.FEAT_DYNAMIC_REAL]
                        if self.use_feat_dynamic_real
                        else []
                    ),
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        return Chain(transforms)
    
    def create_kwargs(self):
        kwargs =self.kwargs.copy()
        kwargs.update(
            len_pred=self.len_pred,
            len_context=self.len_context,
            len_hist=self.len_hist,
            len_infer_model=self.len_infer_model,
            lags_seq=self.lags_seq,
            use_identity=self.use_identity,
            n_sample=self.n_sample,
            scaling=self.scaling,
            len_pred_scale=self.len_pred_scale,
            len_context_scale=self.len_context_scale,
        )
        return kwargs


class NAREstimator(BaseNAREstimator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.train_sampler = FixedNumberInstanceSampler(
            min_past=self.len_hist+self.len_infer_model,
            min_future=0,
        )

    def create_instance_splitter(self, mode):
        instance_sampler = {
            "training": self.train_sampler,
            "validation": self.validation_sampler,
            "test": self.test_sampler,
        }[mode]
        past_length = instance_sampler.min_past

        fields = [
            FieldName.FEAT_AGE,
            FieldName.FEAT_TIME,
            FieldName.OBSERVED_VALUES,
        ]
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=self.len_pred,
            time_series_fields=fields,
        )
    
    def create_training_network(self, device):
        return NARTrainingNetwork(
            **self.create_kwargs()
        ).to(device)

    def create_predictor(
        self,
        transformation,
        trained_network,
        device,
    ):
        prediction_network = NARPredictionNetwork(
            **self.create_kwargs()
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.eval_batch_size,
            freq=self.freq,
            prediction_length=self.len_pred,
            device=device,
        )

    def train_model(
        self,
        training_data,
        validation_data=None,
        num_workers=0,
        prefetch_factor=2,
        shuffle_buffer_length=None,
        cache_data=False,
        **kwargs,
    ):
        transformation = self.create_transformation()

        trained_net = self.create_training_network(self.trainer.device)

        input_names = get_module_forward_input_names(trained_net)

        with env._let(max_idle_transforms=maybe_len(training_data) or 0):
            training_instance_splitter = self.create_instance_splitter("training")

        training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            transform=transformation
            + training_instance_splitter
            + SelectFields(input_names),
            is_train=True,
            cache_data=cache_data,
            cyclic=True,
        )

        training_data_loader = DataLoader(
            training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
            **kwargs,
        )

        n_epochs = self.trainer(
            net=trained_net,
            train_iter=training_data_loader,
            validation_data=validation_data,
            predictor_creator=lambda net: self.create_predictor(
                transformation,
                net,
                self.trainer.device,
            )
        )

        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
            n_epochs=n_epochs,
        )


class NNAREstimator(NAREstimator):
    def __init__(
        self,
        load_path,
        len_total,
        len_infer_post,
        len_lookback,
        n_sample_param,
        n_sample_target,
        **kwargs,
    ):
        super().__init__(
            n_sample=n_sample_target,
            **kwargs,
        )
        self.load_path = load_path
        self.len_total = len_total
        if len_infer_post is None:
            len_infer_post = len_total - self.len_hist
        self.len_infer_post = len_infer_post
        self.len_lookback = len_lookback
        print(f"len_lookback = {len_lookback}")
        self.n_sample_param = n_sample_param

        self.model_train_sampler = FixedNumberInstanceSampler(
            min_past=self.len_hist+self.len_infer_model,
            min_future=0,
        )
        self.posterior_train_sampler = StepWiseInstanceSampler(
            min_past=self.len_hist+self.len_infer_post,
            min_future=0,
            step=self.len_infer_post,
        )

    def create_instance_splitter(self, mode):
        instance_sampler = {
            "posterior_training": self.posterior_train_sampler,
            "model_training": self.model_train_sampler,
            "validation": self.validation_sampler,
            "test": self.test_sampler,
        }[mode]
        past_length = {
            "posterior_training": instance_sampler.min_past,
            "model_training": instance_sampler.min_past,
            "validation": self.len_lookback,
            "test": self.len_lookback,
        }[mode]

        fields = [
            FieldName.FEAT_AGE,
            FieldName.FEAT_TIME,
            FieldName.OBSERVED_VALUES,
        ]
        return InstanceSplitter(
            target_field=FieldName.TARGET,
            is_pad_field=FieldName.IS_PAD,
            start_field=FieldName.START,
            forecast_start_field=FieldName.FORECAST_START,
            instance_sampler=instance_sampler,
            past_length=past_length,
            future_length=self.len_pred,
            time_series_fields=fields,
        )

    def create_kwargs(self):
        kwargs = super().create_kwargs()
        kwargs.update(
            len_pred=self.len_pred,
            len_context=self.len_context,
            len_hist=self.len_hist,
            len_infer_post=self.len_infer_post,
            len_infer_model=self.len_infer_model,
            len_total=self.len_total,
            n_sample_param=self.n_sample_param,
            n_sample_target=self.n_sample,
            lags_seq=self.lags_seq,
            use_identity=self.use_identity,
        )
        return kwargs

    def create_training_network(self, device):
        return NNARTrainingNetwork(
            **self.create_kwargs()
        ).to(device)

    def create_predictor(
        self,
        transformation,
        trained_network,
        device,
    ):
        prediction_network = NNARPredictionNetwork(
            **self.create_kwargs()
        ).to(device)

        copy_parameters(trained_network, prediction_network)
        input_names = get_module_forward_input_names(prediction_network)
        prediction_splitter = self.create_instance_splitter("test")

        return PyTorchPredictor(
            input_transform=transformation + prediction_splitter,
            input_names=input_names,
            prediction_net=prediction_network,
            batch_size=self.eval_batch_size,
            freq=self.freq,
            prediction_length=self.len_pred,
            device=device,
        )

    def train_model(
        self,
        training_data,
        validation_data=None,
        num_workers=0,
        prefetch_factor=2,
        shuffle_buffer_length=None,
        cache_data=False,
        **kwargs,
    ):
        transformation = self.create_transformation()

        trained_net = self.create_training_network(self.trainer.device)

        input_names = get_module_forward_input_names(trained_net)

        with env._let(max_idle_transforms=maybe_len(training_data) or 0):
            posterior_training_instance_splitter = self.create_instance_splitter("posterior_training")
        posterior_training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            transform=transformation
            + posterior_training_instance_splitter
            + SelectFields(input_names),
            is_train=True,
            cache_data=cache_data,
        )

        posterior_training_data_loader = DataLoader(
            posterior_training_iter_dataset,
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
            **kwargs,
        )

        with env._let(max_idle_transforms=maybe_len(training_data) or 0):
            model_training_instance_splitter = self.create_instance_splitter("model_training")

        model_training_iter_dataset = TransformedIterableDataset(
            dataset=training_data,
            transform=transformation
            + model_training_instance_splitter
            + SelectFields(input_names),
            is_train=True,
            cache_data=cache_data,
            cyclic=True,
        )

        model_training_data_loader = DataLoader(
            model_training_iter_dataset,
            batch_size=self.trainer.batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=True,
            worker_init_fn=self._worker_init_fn,
            **kwargs,
        )

        if self.load_path:
            print(f"Loading base model from {self.load_path}")
            base_net = super().create_training_network(self.trainer.device)
            base_net.load_state_dict(torch.load(self.load_path))
            trained_net.load_base_state(base_net.state_dict())
            trained_net.fix_base_params(True)
            del base_net
        else:
            print("Training from scatch")

        self.trainer(
            net=trained_net,
            train_iter=(posterior_training_data_loader, model_training_data_loader),
            validation_data=validation_data,
            predictor_creator=lambda net: self.create_predictor(
                transformation,
                net,
                self.trainer.device,
            )
        )

        return TrainOutput(
            transformation=transformation,
            trained_net=trained_net,
            predictor=self.create_predictor(
                transformation, trained_net, self.trainer.device
            ),
        )

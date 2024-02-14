# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##################################################################################################################
from typing import Optional
from dataclasses import dataclass

from torch import nn
from gluonts.dataset.common import Dataset
from gluonts.dataset.field_names import FieldName
from torch.utils.data import DataLoader
from gluonts.env import env
from gluonts.mx.distribution import (
    MultivariateGaussianOutput,
)
from gluonts.support.util import maybe_len
from gluonts.transform import (
    AddObservedValuesIndicator,
    AddConstFeature,
    AsNumpyArray,
    Chain,
    RemoveFields,
    SelectFields,
    SetField,
    Transformation,
    ExpandDimArray,
    SetFieldIfNotPresent,
    TargetDimIndicator,
)
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deepvar import DeepVAREstimator
from gluonts.dataset.field_names import FieldName
from gluonts.torch.model.predictor import PyTorchPredictor
from pts.model.tempflow import TempFlowEstimator
from pts.model.transformer_tempflow import TransformerTempFlowEstimator
from pts.model.time_grad import TimeGradEstimator
from pts.model.utils import get_module_forward_input_names
from pts.dataset.loader import TransformedIterableDataset


@dataclass
class TrainOutput:
    transformation: Transformation
    trained_net: nn.Module
    predictor: PyTorchPredictor
    n_epochs: int = None


class SimplifiedDeepAREstimator(DeepAREstimator):
    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_static_real:
            remove_field_names.append(FieldName.FEAT_STATIC_REAL)
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)

        return Chain(
            [RemoveFields(field_names=remove_field_names)]
            + (
                [SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0.0])]
                if not self.use_feat_static_cat
                else []
            )
            + (
                [
                    SetField(
                        output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]
                    )
                ]
                if not self.use_feat_static_real
                else []
            )
            + [
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_CAT,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.FEAT_STATIC_REAL,
                    expected_ndim=1,
                    dtype=self.dtype,
                ),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                    dtype=self.dtype,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                    dtype=self.dtype,
                    imputation_method=self.imputation_method,
                ),
                AddConstFeature(
                    output_field=FieldName.FEAT_TIME,
                    target_field=FieldName.TARGET,
                    pred_length=self.prediction_length,
                )
            ]
        )


class SimplifiedDeepVAREstimator(DeepVAREstimator):
    def create_transformation(self) -> Transformation:
        return Chain(
            [
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=1 + len(self.distr_output.event_shape),
                ),
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=0 if self.distr_output.event_shape[0] == 1 else None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddConstFeature(
                    output_field=FieldName.FEAT_TIME,
                    target_field=FieldName.TARGET,
                    pred_length=self.prediction_length,
                ),
                SetFieldIfNotPresent(
                    field=FieldName.FEAT_STATIC_CAT, value=[0.0]
                ),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )


class DiagonalMultivariateGaussianOutput(MultivariateGaussianOutput):
    def domain_map(self, F, mu_vector, L_vector):
        d = self.dim
        L_matrix = L_vector.reshape((-2, d, d, -4), reverse=1)
        L_diag = F.broadcast_mul(
            F.Activation(
                F.broadcast_mul(L_matrix, F.eye(d)), act_type="softrelu"
            ),
            F.eye(d),
        )
        return mu_vector, L_diag

class ValTempFlowEstimator(TempFlowEstimator):
    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) :
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
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
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
            n_epochs = n_epochs,
        )


class ValTransformerTempFlowEstimator(TransformerTempFlowEstimator):
    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) :
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
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
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
            n_epochs = n_epochs,
        )


class SimplifiedTransformerTempFlowEstimator(TransformerTempFlowEstimator):
    def create_transformation(self) -> Transformation:
        remove_field_names = [FieldName.FEAT_DYNAMIC_CAT]
        if not self.use_feat_dynamic_real:
            remove_field_names.append(FieldName.FEAT_DYNAMIC_REAL)
        
        return Chain(
            [
                RemoveFields(field_names=remove_field_names),
                AsNumpyArray(
                    field=FieldName.TARGET,
                    expected_ndim=2,
                ),
                ExpandDimArray(
                    field=FieldName.TARGET,
                    axis=None,
                ),
                AddObservedValuesIndicator(
                    target_field=FieldName.TARGET,
                    output_field=FieldName.OBSERVED_VALUES,
                ),
                AddConstFeature(
                    output_field=FieldName.FEAT_TIME,
                    target_field=FieldName.TARGET,
                    pred_length=self.prediction_length,
                ),
                SetFieldIfNotPresent(field=FieldName.FEAT_STATIC_CAT, value=[0]),
                TargetDimIndicator(
                    field_name="target_dimension_indicator",
                    target_field=FieldName.TARGET,
                ),
                AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1),
            ]
        )


class ValTimeGradEstimator(TimeGradEstimator):
    def train_model(
        self,
        training_data: Dataset,
        validation_data: Optional[Dataset] = None,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        shuffle_buffer_length: Optional[int] = None,
        cache_data: bool = False,
        **kwargs,
    ) :
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
            shuffle_buffer_length=shuffle_buffer_length,
            cache_data=cache_data,
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
            n_epochs = n_epochs,
        )


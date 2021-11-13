import os
import pytest
import numpy as np
import tensorflow as tf
from dataset.dataset_echonet import EchoNetDataset
from dataset.dataset_generator import DatasetGenerator
from echotrain.model.loss import iou_coef_loss, dice_coef_loss, soft_dice_loss, soft_iou_loss
from utils import load_config_file


class TestClass:

    @pytest.fixture
    def tensor_data_test_case(self):
        """Returns 1*3*3 tensors"""

        y_true = np.array([[1., 1., 1.], [1., 0., 1.], [0., 0., 0.]])
        y_pred = np.array([[0.6, 0.9, 0.7], [0.8, 0.1, 0.8], [0.1, 0.2, 0.3]])
        y_true, y_pred = np.expand_dims(y_true, -1), np.expand_dims(y_pred, -1)

        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        return y_true, y_pred

    def test_iou_loss_coef(self, tensor_data_test_case):
        y_true, y_pred = tensor_data_test_case
        loss = iou_coef_loss(y_true, y_pred)
        print('iou_coef_loss:', loss)

        assert loss <= 1
        assert loss.shape == ()
        assert 'tensor' in str(type(loss))
        assert 'float' in str(loss.dtype)

    def test_dice_loss_coef(self, tensor_data_test_case):
        y_true, y_pred = tensor_data_test_case
        loss = dice_coef_loss(y_true, y_pred)
        print('dice_coef_loss:', loss)

        assert loss <= 1
        assert loss.shape == ()
        assert 'tensor' in str(type(loss))
        assert 'float' in str(loss.dtype)

    def test_soft_dice_loss(self, tensor_data_test_case):
        y_true, y_pred = tensor_data_test_case
        loss = soft_dice_loss(y_true, y_pred)
        print('soft_dice_loss:', loss)

        assert loss <= 1
        assert loss.shape == ()
        assert 'tensor' in str(type(loss))
        assert 'float' in str(loss.dtype)

    def test_soft_iou_loss(self, tensor_data_test_case):
        y_true, y_pred = tensor_data_test_case
        loss = soft_iou_loss(y_true, y_pred)
        print('soft_iou_loss:', loss)

        assert loss <= 1
        assert loss.shape == ()
        assert 'tensor' in str(type(loss))
        assert 'float' in str(loss.dtype)







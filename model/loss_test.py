import os
import pytest
import numpy as np
import tensorflow as tf
from dataset.dataset_echonet import EchoNetDataset
from dataset.dataset_generator import DatasetGenerator
from .loss import iou_coef_loss, dice_coef_loss, soft_dice_loss, soft_iou_loss
from utils import load_config_file


class TestClass:

    @pytest.fixture
    def config(self):
        config_path = "./config/config_example_echonet.yaml"
        config = load_config_file(os.path.abspath(config_path))
        return config

    @pytest.fixture
    def dataset(self, config):
        dataset = EchoNetDataset(config)
        return dataset

    @pytest.fixture
    def generator_inputs(self, dataset):
        instance = {
            'list_images_dir': dataset.x_train_dir,
            'list_labels_dir': dataset.y_train_dir,
            'batch_size': dataset.batch_size,
            'input_size': dataset.input_size,
            'n_channels': dataset.n_channels,
            'channel_last': True,
            'to_fit': dataset.to_fit,
            'shuffle': dataset.shuffle,
            'seed': dataset.seed
        }
        return instance

    @pytest.fixture
    def data_gen(self, generator_inputs):
        data_gen = DatasetGenerator(**generator_inputs)
        return data_gen

    @pytest.fixture
    def tensor_data_test_case(self):
        """Returns 3*3*1 tensors"""

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







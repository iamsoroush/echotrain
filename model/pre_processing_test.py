import os
import pytest
import numpy as np
from dataset.dataset_echonet import EchoNetDataset
from dataset.dataset_generator import DatasetGenerator
from .pre_processing import PreProcessor
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
    def pre_processor(self, config):
        pre_processor = PreProcessor(config)
        return pre_processor

    def test_img_preprocess(self, pre_processor, data_gen):
        img = data_gen.next()[0][0]
        pre_processed_img = pre_processor.img_preprocess(img)

        # Resizing
        assert pre_processed_img.shape == (data_gen.input_size[0], data_gen.input_size[1], 1)

        # Type checking
        assert 'float' in str(pre_processed_img.dtype)

        # Normalization
        assert 0 <= pre_processed_img.all() <= 1

    def test_batch_preprocess(self, pre_processor, data_gen):
        batch = data_gen.next()

        pre_processed_batch = pre_processor.batch_preprocess(batch)

        assert len(pre_processed_batch[0]) == data_gen.batch_size

        assert pre_processed_batch[0][0].shape == (data_gen.input_size[0], data_gen.input_size[1], 1)

        # Type checking
        assert 'float' in str(pre_processed_batch[0][0].dtype)

        # Normalization
        assert 0 <= pre_processed_batch[0][0].all() <= 1

    @pytest.mark.parametrize("add_augmentation", [
        True,
        False,
    ])
    def test_add_preprocess(self, pre_processor, data_gen, add_augmentation):
        pre_processed_gen = pre_processor.add_preprocess(data_gen, add_augmentation=add_augmentation)

        assert 'generator' in str(type(pre_processed_gen))

        pre_processed_batch = next(pre_processed_gen)

        assert pre_processed_batch[0][0].shape == (data_gen.input_size[0], data_gen.input_size[1], 1)

        # Type checking
        assert 'float' in str(pre_processed_batch[0][0].dtype)

        # Normalization
        assert 0 <= pre_processed_batch[0][0].all() <= 1






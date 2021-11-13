import os
import pytest
import numpy as np
from dataset.dataset_echonet import EchoNetDataset
from dataset.dataset_generator import DatasetGenerator
from echotrain.model.augmentation import Augmentation
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
    def augmentor(self, config):
        augmentor = Augmentation(config)
        return augmentor

    def test_batch_augmentation(self, augmentor, data_gen):
        batch = data_gen.next()

        augmented_batch = augmentor.batch_augmentation(batch)

        assert len(augmented_batch[0]) == data_gen.batch_size

        assert augmented_batch[0][0].shape == (112, 112, 1)

        # Type checking
        assert 'float' in str(augmented_batch[0][0].dtype)

    def test_add_augmentation(self, augmentor, data_gen):
        augmented_gen = augmentor.add_augmentation(data_gen)

        assert 'generator' in str(type(augmented_gen))

        augmented_batch = next(augmented_gen)

        assert augmented_batch[0][0].shape == (112, 112, 1)

        # Type checking
        assert 'float' in str(augmented_batch[0][0].dtype)






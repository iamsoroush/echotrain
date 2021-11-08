import pytest
import numpy as np
import os
import sys
import skimage.io as io
sys.path.append(os.path.abspath('../echotrain'))
from .dataset_echonet import EchoNetDataset
from .dataset_camus import CAMUSDataset
from .dataset_generator import DatasetGenerator
from utils import load_config_file


class TestClass:

    @pytest.fixture
    def config(self, path="./config/config_example_echonet.yaml"):
        config_path = path
        config = load_config_file(os.path.abspath(config_path))
        return config

    @pytest.fixture
    def echonet_dataset(self, config):
        dataset = EchoNetDataset(config)
        return dataset

    @pytest.fixture
    def camus_dataset(self, config):
        dataset = CAMUSDataset(config(path="./config/config_example.yaml"))
        return dataset

    @pytest.fixture
    def sample_input(self, echonet_dataset):
        instance = {
            'list_images_dir': echonet_dataset.x_train_dir,
            'list_labels_dir': echonet_dataset.y_train_dir,
            'batch_size': echonet_dataset.batch_size,
            'input_size':  echonet_dataset.input_size,
            'n_channels': echonet_dataset.n_channels,
            'channel_last': True,
            'to_fit': True,
            'shuffle': True,
            'seed': None
        }
        return instance

    def test_get_n_iter(self, echonet_dataset, sample_input):
        data_gen = DatasetGenerator(**sample_input)
        assert data_gen.get_n_iter() == len(echonet_dataset.x_train_dir)/echonet_dataset.batch_size

    def test_generate_x(self, echonet_dataset, sample_input, config):
        data_gen = DatasetGenerator(**sample_input)
        sample_img_dir = np.random.choice(data_gen.list_images_dir)
        print(sample_img_dir)
        sample_img = data_gen.generate_x(sample_img_dir)
        assert sample_img == (112, 112)

    def test_generate_y(self, echonet_dataset, sample_input):
        data_gen = DatasetGenerator(**sample_input)
        assert data_gen.get_n_iter() == len(echonet_dataset.x_train_dir)/echonet_dataset.batch_size






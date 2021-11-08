import sys
import os
import warnings
utils_dir = os.path.abspath('../echotrain/')
sys.path.append(utils_dir)
from dataset.dataset_echonet import EchoNetDataset
# from handling_yaml import load_config_file
from utils import load_config_file
import pytest

# warnings.filterwarnings("ignore", category=DeprecationWarning)


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

    @pytest.mark.parametrize("x, y", [
        ([1, 2, 3], {1: 4, 2: 5, 3: 6}),
    ])
    def test_shuffle_func(self, dataset, x, y):
        shuffled_x, shuffled_y = dataset._shuffle_func(x, y)

        # Testing if both data and labels are shuffled the same.
        assert shuffled_x == list(shuffled_y.keys())

    def test_create_data_generators(self, dataset):
        train_gen, val_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

        # Testing if the number of iterations are equal to the real numbers as in the echonet dataset
        assert n_iter_train == len(dataset.x_train_dir) / dataset.batch_size
        assert n_iter_val == len(dataset.x_val_dir) / dataset.batch_size

    def test_create_test_data_generator(self, dataset):
        test_gen, n_iter_test = dataset.create_test_data_generator()

        # Testing if the number of iterations are equal to the real numbers as in the echonet dataset
        assert n_iter_test == len(dataset.x_test_dir) / dataset.batch_size


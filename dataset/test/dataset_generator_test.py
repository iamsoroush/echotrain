import pytest
import numpy as np
import os
import skimage.io as io
from echotrain.dataset.dataset_echonet import EchoNetDataset
from echotrain.dataset.dataset_camus import CAMUSDataset
from echotrain.dataset.dataset_generator import DatasetGenerator
from echotrain.utils import load_config_file


class TestClass:

    @pytest.fixture
    def config(self):
        root_dir = os.path.abspath(os.curdir)
        if 'echotrain' not in root_dir:
            root_dir = os.path.join(root_dir, 'echotrian').replace('\\', '/')
        config_path = os.path.join(root_dir, "config/config_example_echonet.yaml")
        config = load_config_file(config_path)
        return config

    # @pytest.fixture
    # def cmaus(self, config):
    #     dataset = CAMUSDataset(config(config_path="./config/config_example.yaml"))
    #     return dataset

    @pytest.fixture
    def echonet(self, config):
        dataset = EchoNetDataset(config)
        return dataset

    @pytest.fixture
    def generator_inputs(self, echonet):
        instance = {
            'list_images_dir': echonet.x_train_dir,
            'list_labels_dir': echonet.y_train_dir,
            'batch_size': echonet.batch_size,
            'input_size': echonet.input_size,
            'n_channels': echonet.n_channels,
            'channel_last': True,
            'to_fit': echonet.to_fit,
            'shuffle': echonet.shuffle,
            'seed': echonet.seed
        }
        return instance

    @pytest.fixture
    def data_gen(self, generator_inputs):
        data_gen = DatasetGenerator(**generator_inputs)
        return data_gen

    def test_get_n_iter(self, data_gen, echonet):
        assert data_gen.get_n_iter() == len(echonet.x_train_dir)/echonet.batch_size

    def test_generate_x(self, data_gen):
        sample_img_dir = np.random.choice(data_gen.list_images_dir)
        sample_img = data_gen.generate_x([sample_img_dir])[0]

        # Testing the shape of an input image read by generate_x
        assert sample_img.shape == (112, 112, 1)

    def test_generate_y(self, data_gen):
        seed = data_gen.seed
        sample_img_dir = str(np.random.RandomState(seed).choice(data_gen.list_images_dir))
        sample_lbl_dir = str(data_gen.list_labels_dir[sample_img_dir])

        # Testing if the image and its label match together
        assert sample_img_dir.split('/')[-1].split('.')[0] in sample_lbl_dir.split('/')[-1].split('.')[0]
        sample_lbl = data_gen.generate_y([sample_img_dir])[0]

        # reading the label manually to compare thoose two
        y_4ch = io.imread(sample_lbl_dir, plugin='simpleitk')

        # extract just left ventricle label from y_4ch_cat
        y_4ch_lv = np.where(y_4ch == 1, 1, 0)

        # Testing the shape of an input image read by generate_x
        assert sample_lbl.shape == (112, 112)

        # Testing the values
        assert sample_lbl.all() == y_4ch_lv.all()

    def test_class(self, data_gen):

        # Type Checking
        assert 'DatasetGenerator' in str(type(data_gen))

        batch = data_gen.next()

        # batch size checking
        assert len(batch[0]) == data_gen.batch_size

        # dtype checking
        assert 'numpy.ndarray' in str(type(batch[0][0]))

        # value range checking
        assert 0 <= batch[0][0].all() <= 255


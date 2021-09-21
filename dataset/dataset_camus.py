# requirements

from dataset_generator import DatasetGenerator
from dataset_base import DatasetBase
from glob import glob  # for listing the directory of dataset
import numpy as np
import os
import random


class CAMUSDataset(DatasetBase):
    """
    This class makes our dataset ready to use by given desired values to its parameters
    and by calling the "create_data_generators" or "create_test_data_generator" function,
    reads the data from the given directory as follow:

    HOW TO:
    dataset = CAMUSDataset(config.data_handler)

    # for training set:
    train_gen, val_gen, n_iter_train, n_iter_val= dataset.create_data_generators(dataset_dir)

    # for test set:
    dataset_gen = dataset.create_test_generator(test_set_dir)
    """

    def __init__(self, config):
        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        batch_size: batch size, int
        input_size: input image resolution, (h, w)
        n_channels: number of channels, int
        split_ratio: ratio of splitting into train and validation, float
        to_fit: for predicting time, bool
        shuffle: if True the dataset will shuffle with random_state of seed, bool
        seed: seed, int
        // changing from "input_res" to "input_size"
        """

        super(CAMUSDataset, self).__init__(config)
        self.batch_size = config.data_handler.batch_size
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.input_size = (self.input_h, self.input_w)
        self.n_channels = config.n_channels
        self.split_ratio = config.data_handler.split_ratio
        self.seed = config.data_handler.seed
        self.shuffle = config.data_handler.shuffle
        self.to_fit = config.data_handler.to_fit
        self.dataset_dir = config.data_handler.dataset_dir

    def create_data_generators(self):

        """Creates data generators based on batch_size, input_size

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """
        # dataset_dir = self.dataset_dir

        list_images_dir, list_labels_dir = self.fetch_data()

        # shuffling
        if self.shuffle:
            list_images_dir, list_labels_dir = self.shuffle_func(list_images_dir,
                                                                 list_labels_dir)
        # splitting
        x_train_dir, y_train_dir, x_val_dir, y_val_dir = self.split(list_images_dir,
                                                                    list_labels_dir,
                                                                    self.split_ratio)

        train_data_gen = DatasetGenerator(x_train_dir, y_train_dir, self.batch_size,
                                          self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)
        val_data_gen = DatasetGenerator(x_val_dir, y_val_dir, self.batch_size,
                                        self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)

        n_iter_train = train_data_gen.get_n_iter()
        n_iter_val = val_data_gen.get_n_iter()

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

    def create_test_data_generator(self):
        """
        Creates data generators based on batch_size, input_size

        :returns dataset_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_dataset: number of iterations per epoch for train_data_gen
        """
        list_images_dir, list_labels_dir = self.fetch_data()

        dataset_gen = DatasetGenerator(list_images_dir, list_labels_dir, self.batch_size,
                                       self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)

        n_iter_dataset = dataset_gen.get_n_iter()

        return dataset_gen, n_iter_dataset

    def fetch_data(self):
        """
        fetches data from directory of A4C view images of CAMUS dataset

        dataset_dir: directory address of the dataset

        :return list_images_dir: list of the A4C view images directory
        :return list_labels_dir: list of the type_map labels directory
        """

        dataset_dir = self.dataset_dir

        # Directory list of the A4C view of the ED ( End Diastole ) frame images.
        # x_4ch_ed_dir: list[str]
        # y_4ch_ed_dir: list[str]
        x_4ch_ed_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ED.mhd'))  # images directory
        y_4ch_ed_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ED_gt.mhd'))  # segmentation labels directory

        # Directory list of the A4C view of the ES ( End Systole ) frame images.
        # x_4ch_es_dir: list[str]
        # y_4ch_es_dir: list[str]
        x_4ch_es_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ES.mhd'))  # images directory
        y_4ch_es_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ES_gt.mhd'))  # segmentation labels directory

        # Concatenating ES and ED images and labels
        x_4ch_dir = np.concatenate((x_4ch_ed_dir, x_4ch_es_dir), axis=0)
        y_4ch_dir = np.concatenate((y_4ch_ed_dir, y_4ch_es_dir), axis=0)

        list_images_dir = x_4ch_dir
        list_labels_dir = {}
        for i in range(len(y_4ch_dir)):
            list_labels_dir[x_4ch_dir[i]] = y_4ch_dir[i]

        return list_images_dir, list_labels_dir

    def shuffle_func(self, x, y):

        """
        makes a shuffle index array to make a fixed shuffling order for both X, y

        :param x: list of images, np.ndarray
        :param y: list of segmentation labels, np.ndarray

        :return x: shuffled list of images, np.ndarray
        :return y: shuffled list of segmentation labels, np.ndarray
        """

        # seed initialization
        if self.seed is None:
            seed = random.Random(None).getstate()
        else:
            seed = self.seed

        # shuffling
        y_list = list(y.items())
        random.Random(seed).shuffle(x)
        random.Random(seed).shuffle(y_list)
        y = dict(y_list)
        return x, y

    @staticmethod
    def split(x, y, split_ratio):

        """
        splits the dataset into train and validation set by the corresponding ratio
        the ratio is "train portion/whole data"

        :param x: list of images, np.ndarray
        :param y: list of segmentation labels, np.ndarray
        :param split_ratio: split ratio for trainset, float

        :return x_train: images train_set, np.ndarray
        :return y_train: segmentation labels train_set, np.ndarray
        :return x_val: images validation_set, np.ndarray
        :return y_val: segmentation labels validation_set, np.ndarray
        """
        # set train size by split_ratio var
        train_size = round(len(x) * split_ratio)
        # splitting
        x_train = x[:train_size]
        y_train = dict(list(y.items())[:train_size])

        x_val = x[train_size:]
        y_val = dict(list(y.items())[train_size:])

        return x_train, y_train, x_val, y_val




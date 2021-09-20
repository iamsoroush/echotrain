# requirements

from dataset_generator import DatasetGenerator
from dataset_base import DatasetBase
from glob import glob  # for listing the directory of dataset
import numpy as np
import os
import random


class CAMUSDataset(DatasetBase):
    """
    HOW TO:
    train_gen, val_gen, n_iter_train, n_iter_val= CAMUSDataset(batch_size, input_size, n_channels, split_ratio,
                                                               seed=None, shuffle=True, to_fit=True,
                                                               config=None).create_data_generators(dataset_dir)
    """

    def __init__(self, batch_size, input_size, n_channels, split_ratio, seed=None,
                 shuffle=True, to_fit=True, config=None):
        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        :param batch_size: batch size, int
        :param input_size: input image resolution, (h, w)
        :param n_channels: number of channels, int
        :param split_ratio: ratio of splitting into train and validation, float
        :param to_fit: for predicting time, bool
        :param shuffle: if True the dataset will shuffle with random_state of seed, bool
        :param seed: seed, int
        // changing from "input_res" to "input_size"
        """

        super(CAMUSDataset, self).__init__(config)
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_channels = n_channels
        self.split_ratio = split_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.to_fit = to_fit

    def create_data_generators(self, dataset_dir):

        """Creates data generators based on batch_size, input_size

        :param dataset_dir: dataset directory

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """
        list_images_dir, list_labels_dir = self.fetch_data(dataset_dir)

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
        n_iter_train = train_data_gen.__len__()
        n_iter_val = val_data_gen.__len__()

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

    def test_create_data_generator(self, dataset_dir):
        """
        Creates data generators based on batch_size, input_size

        :param dataset_dir: dataset directory

        :returns dataset_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_dataset: number of iterations per epoch for train_data_gen
        """
        list_images_dir, list_labels_dir = self.fetch_data(dataset_dir)

        dataset_gen = DatasetGenerator(list_images_dir, list_labels_dir, self.batch_size,
                                       self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)

        n_iter_dataset = dataset_gen.__len__()

        return dataset_gen, n_iter_dataset

    @staticmethod
    def fetch_data(dataset_dir):
        """
        fetches data from directory of A4C view images of CAMUS dataset

        :param dataset_dir: directory address of the dataset

        :return list_images_dir: list of the A4C view images directory
        :return list_labels_dir: list of the type_map labels directory
        """

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

#
# # from config import config_example
# if __name__ == '__main__':
#     # config_path = "D:/AIMedic/FinalProject_echocardiogram/echoC_Codes/main/echotrain/config"
#     dataset_dir = "D:/AIMedic/FinalProject_echocardiogram/echoC_Dataset/CAMUS/training"
#     dataset = CAMUSDataset(batch_size=10, input_size=(128, 128), n_channels=1, split_ratio=8 / 9)
#     train_gen, val_gen, n_iter_train, n_iter_val = dataset.create_data_generators(dataset_dir)
#     train_gen.random_visualization()

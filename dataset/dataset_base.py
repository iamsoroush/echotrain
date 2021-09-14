# requirements

from glob import glob  # for listing the directory of dataset
import skimage.io as io  # to read the .mhd and .raw data
import SimpleITK  # plugin for skimage.io
import numpy as np
import tensorflow as tf
import os


# dataset_dir="D:/AIMedic/FinalProject_echocardiogram/echoC_Dataset/CAMUS/training/"

class DatasetBase:

    def __init__(self, dataset_dir,
                 batch_size, input_size, n_channels, ):
        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        :param n_channels: number of channels, int
        :param batch_size: batch size, int
        :param input_size: input image resolution, (h, w)
        // changing from "input_res" to "input_size"
        """

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_channels = n_channels

    def fetch_data(self):
        """
        fetching data from directory of A4C view images of CAMUS dataset

        :param dataset_dir: directory address of the dataset

        :return data_list: list of the A4C view images
        :return labels: list of the type_map labels
        """
        ######### END-DIASTOLE #########
        # Directory list of the A4C view of the ED ( End Diastole ) frame images.
        # X_4CH_ED_dir: list[str]
        # y_4CH_ED_dir: list[str]
        X_4CH_ED_dir = glob(os.path.join(self.dataset_dir, '*/*_4CH_ED.mhd'))  # images directory
        y_4CH_ED_dir = glob(os.path.join(self.dataset_dir, '*/*_4CH_ED_gt.mhd'))  # segmentation labels directory

        # fetching the data from directory and reading images of .mhd format with the help of SimpleITK plugin,
        # and makes all of them channel last order.
        # X_4CH_ED: list[numpy.ndarray]
        # y_4CH_ED: list[numpy.ndarray]
        X_4CH_ED = list(map(lambda x: np.moveaxis(io.imread(x, plugin='simpleitk'), 0, -1), X_4CH_ED_dir))
        y_4CH_ED = list(map(lambda x: np.moveaxis(io.imread(x, plugin='simpleitk'), 0, -1), y_4CH_ED_dir))

        # resizing all data into the input_size ( target_size ) dimensions.
        # X_4CH_ED_resized: list[numpy.ndarray]
        # y_4CH_ED_resized: list[numpy.ndarray]
        X_4CH_ED_resized = np.array(list(map(lambda x: tf.image.resize(x,
                                                                       self.input_size,
                                                                       antialias=False,
                                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                             X_4CH_ED)))
        y_4CH_ED_resized = np.array(list(map(lambda x: tf.image.resize(x,
                                                                       self.input_size,
                                                                       antialias=False,
                                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                             y_4CH_ED)))

        ######### END-SYSTOLE #########
        # Directory list of the A4C view of the ES ( End Systole ) frame images.
        # X_4CH_ES_dir: list[str]
        # y_4CH_ES_dir: list[str]
        X_4CH_ES_dir = glob(os.path.join(self.dataset_dir, '*/*_4CH_ES.mhd'))
        y_4CH_ES_dir = glob(os.path.join(self.dataset_dir, '*/*_4CH_ES_gt.mhd'))

        # fetching the data from directory and reading images of .mhd format with the help of SimpleITK plugin
        # and makes all of them channel last order.
        # X_4CH_ES: list[numpy.ndarray]
        # y_4CH_ES: list[numpy.ndarray]
        X_4CH_ES = list(map(lambda x: np.moveaxis(io.imread(x, plugin='simpleitk'), 0, -1), X_4CH_ES_dir))
        y_4CH_ES = list(map(lambda x: np.moveaxis(io.imread(x, plugin='simpleitk'), 0, -1), y_4CH_ES_dir))

        # resizing all data into the input_size ( target_size ) dimensions.
        # X_4CH_ES_resized: list[numpy.ndarray]
        # y_4CH_ES_resized: list[numpy.ndarray]
        X_4CH_ES_resized = np.array(list(map(lambda x: tf.image.resize(x,
                                                                       self.input_size,
                                                                       antialias=False,
                                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                             X_4CH_ES)))
        y_4CH_ES_resized = np.array(list(map(lambda x: tf.image.resize(x,
                                                                       self.input_size,
                                                                       antialias=False,
                                                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR),
                                             y_4CH_ES)))

        dataset = np.concatenate((X_4CH_ED_resized, X_4CH_ES_resized), axis=0)
        labels = np.concatenate((y_4CH_ED_resized, y_4CH_ES_resized), axis=0)
        return dataset, labels

    def create_data_generators(self, dataset_dir, ):
        """Creates data generators based on batch_size, input_size

        :param dataset_dir: dataset's directory

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """

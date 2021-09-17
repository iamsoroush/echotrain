# requirements

from glob import glob  # for listing the directory of dataset
import skimage.io as io  # to read the .mhd and .raw data
import SimpleITK  # plugin for skimage.io
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import os


class DatasetGenerator(tf.keras.utils.Sequence):
    def __init__(self, list_images_dir, list_labels_dir,
                 batch_size, input_size, n_channels, to_fit=True, shuffle=True, seed=None):
        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        :param list_images_dir: list of images directory, list
        :param list_labels_dir: segmentation labels directory as values with the list_images_dir as keys, dict
        :param batch_size: batch size, int
        :param input_size: input image resolution, (h, w)
        :param n_channels: number of channels, int
        :param to_fit: for predicting time, bool
        :param shuffle: if True the dataset will shuffle with random_state of seed, bool
        :param seed: seed, int
        // changing from "input_res" to "input_size"
        """

        self.list_images_dir = list_images_dir
        self.list_labels_dir = list_labels_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_channels = n_channels
        self.seed = seed
        self.shuffle = shuffle
        self.to_fit = to_fit
        self.on_epoch_end()

    def __getitem__(self, index):
        """Generate one batch of data

        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of the data
        batch_dir_list = [self.list_images_dir[k] for k in indexes]

        # Generate data
        X = self.generate_X(batch_dir_list)

        if self.to_fit:
            y = self.generate_y(batch_dir_list)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        """shuffles indexes after each epoch
        """
        # ????????
        self.indexes = np.arange(len(self.list_images_dir))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(np.floor(len(self.list_images_dir) / self.batch_size))

    def generate_X(self, batch_dir_list):
        """
        reads A4C view images of CAMUS dataset

        :param batch_dir_list

        :return X_4CH_preprocessed: array of preprocessed images
        """

        # reading images of .mhd format with the help of SimpleITK plugin,
        # and makes all of them channel last order.
        # X_4CH_ED: list[numpy.ndarray]
        X_4CH = list(map(lambda x: np.moveaxis(io.imread(x, plugin='simpleitk'), 0, -1),
                         batch_dir_list))

        # Pre-processing labels
        # X_4CH_ED_resized: list[numpy.ndarray]
        X_4CH_preprocessed = np.array(list(map(self.pre_process, X_4CH))) / 255.

        return X_4CH_preprocessed

    def generate_y(self, batch_dir_list):
        """
        reads A4C view segmentation labels of CAMUS dataset

        :param batch_dir_list

        :return y_4CH_preprocessed: array of preprocessed images
        """
        # reading segmentation labels of .mhd format with the help of SimpleITK plugin,
        # and makes all of them channel last order.
        # y_4CH: list[numpy.ndarray]
        y_4CH = list(map(lambda x: np.moveaxis(io.imread(x, plugin='simpleitk'), 0, -1),
                         [self.list_labels_dir[image_path] for image_path in batch_dir_list]))

        # Pre-processing labels
        # X_4CH_ED_resized: list[numpy.ndarray]
        y_4CH_preprocessed = np.array(list(map(self.pre_process, y_4CH)))

        #to categorical
        y_4CH_preprocessed=to_categorical(y_4CH_preprocessed)

        return y_4CH_preprocessed

    def pre_process(self, image):
        """
        pre-processes images mentioned by the user

        :param image: input image, np.ndarray

        :return image_pre_processed: pre-processed image, np.ndarray
        """
        # resizing image into the input_size ( target_size ) dimensions.
        image_resized = tf.image.resize(image,
                                        self.input_size,
                                        antialias=False,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        image_pre_processed = image_resized
        return image_pre_processed

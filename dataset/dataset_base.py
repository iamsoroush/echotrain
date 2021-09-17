# requirements


from dataset_generator import DatasetGenerator
from glob import glob  # for listing the directory of dataset
import skimage.io as io  # to read the .mhd and .raw data
import SimpleITK  # plugin for skimage.io
import numpy as np
import tensorflow as tf
import os
import random
import segmentation_models as sm
from segmentation_models import Unet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
sm.set_framework("tf.keras")



# dataset_dir="D:/AIMedic/FinalProject_echocardiogram/echoC_Dataset/CAMUS/training/"

class DatasetBase:

    def __init__(self, dataset_dir,batch_size, input_size, n_channels, split_ratio=1, to_fit=True, shuffle=True, seed=None):
        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        :param dataset_dir: dataset directory address, str
        :param list_images_dir: list of images directory, list
        :param list_labels_dir: segmentation labels directory as values with the list_images_dir as keys, dict
        :param batch_size: batch size, int
        :param input_size: input image resolution, (h, w)
        :param n_channels: number of channels, int
        :param split_ratio: ratio of splitting into train and validation, float
        :param to_fit: for predicting time, bool
        :param shuffle: if True the dataset will shuffle with random_state of seed, bool
        :param seed: seed, int
        // changing from "input_res" to "input_size"
        """

        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.n_channels = n_channels
        self.split_ratio = split_ratio
        self.seed = seed
        self.shuffle = shuffle
        self.to_fit = to_fit


    def create_data_generators(self):
        """Creates data generators based on batch_size, input_size

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """
        list_images_dir, list_labels_dir=self.fetch_data(self.dataset_dir)

        # shuffling
        if self.shuffle:
            list_images_dir, list_labels_dir = self.unison_shuffle(list_images_dir,
                                                                    list_labels_dir)
        # splitting
        if self.split_ratio != 1:
            X_train_dir, y_train_dir, X_val_dir, y_val_dir = self.split(list_images_dir,
                                                                        list_labels_dir,
                                                                        self.split_ratio)
            train_data_gen = DatasetGenerator(X_train_dir, y_train_dir, self.batch_size,
                                              self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)
            val_data_gen = DatasetGenerator(X_val_dir, y_val_dir, self.batch_size,
                                            self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)
        else:
            raise Exception('there is no data for not splitting part')

        print("The iteration of the training data is:")
        print(train_data_gen.__len__())
        print("The iteration of the validation data is:")
        print(val_data_gen.__len__())

        return train_data_gen, val_data_gen

    def fetch_data(self, dataset_dir):
        """
        fetches data from directory of A4C view images of CAMUS dataset

        :param dataset_dir: directory address of the dataset

        :return list_images_dir: list of the A4C view images directory
        :return list_labels_dir: list of the type_map labels directory
        """

        # Directory list of the A4C view of the ED ( End Diastole ) frame images.
        # X_4CH_ED_dir: list[str]
        # y_4CH_ED_dir: list[str]
        X_4CH_ED_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ED.mhd'))  # images directory
        y_4CH_ED_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ED_gt.mhd'))  # segmentation labels directory

        # Directory list of the A4C view of the ES ( End Systole ) frame images.
        # X_4CH_ES_dir: list[str]
        # y_4CH_ES_dir: list[str]
        X_4CH_ES_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ES.mhd'))  # images directory
        y_4CH_ES_dir = glob(os.path.join(dataset_dir, '*/*_4CH_ES_gt.mhd'))  # segmentation labels directory

        # Concatenating ES and ED images and labels
        X_4CH_dir = np.concatenate((X_4CH_ED_dir, X_4CH_ES_dir), axis=0)
        y_4CH_dir = np.concatenate((y_4CH_ED_dir, y_4CH_ES_dir), axis=0)

        list_images_dir = X_4CH_dir
        list_labels_dir={}
        for i in range(len(y_4CH_dir)):
            list_labels_dir[X_4CH_dir[i]] = y_4CH_dir[i]

        return list_images_dir, list_labels_dir

    def unison_shuffle(self, x, y):
        """
        makes a shuffle index array to make a fixed shuffling order for both X, y

        :param x: list of images, np.ndarray
        :param y: list of segmentation labels, np.ndarray

        :return x: shuffled list of images, np.ndarray
        :return y: shuffled list of segmentation labels, np.ndarray
        """

        seed = 101
        y_list = list(y.items())
        random.Random(seed).shuffle(x)
        random.Random(seed).shuffle(y_list)
        y = dict(y_list)
        return x, y

    def split(self, x, y, split_ratio):
        """
        splits the dataset into train and validation set by the corresponding ratio
        the ratio is "train portion/whole data"

        :param x: list of images, np.ndarray
        :param y: list of segmentation labels, np.ndarray
        :param split_ratio: split ratio for trainset, float

        :return X_train: images train_set, np.ndarray
        :return y_train: segmentation labels train_set, np.ndarray
        :return X_val: images validation_set, np.ndarray
        :return y_val: segmentation labels validation_set, np.ndarray
        """
        train_size = round(len(x) * split_ratio)

        X_train = x[:train_size]
        y_train = dict(list(y.items())[:train_size])

        X_val = x[train_size:]
        y_val = dict(list(y.items())[train_size:])

        return X_train, y_train, X_val, y_val

    def random_visualization(self):

        raise Exception('not implemented')


'''
        ######### END-DIASTOLE #########
        # fetching the data from directory and reading images of .mhd format with the help of SimpleITK plugin,
        # and makes all of them channel last order.
        # X_4CH_ED: list[numpy.ndarray]
        # y_4CH_ED: list[numpy.ndarray]
        X_4CH = list(map(lambda x: np.moveaxis(io.imread(x, plugin='simpleitk'), 0, -1), X_4CH_ED_dir))
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
'''

train_gen, val_gen = DatasetBase('D:/training/', batch_size = 10, input_size=(128, 128),shuffle=True, n_channels=1, split_ratio =8/9).create_data_generators()

print(train_gen.__len__())
print(val_gen.__len__())

model = Unet(
            'resnet34',
            input_shape=(128,128,1),
            classes = 4,
            activation='softmax',
            encoder_freeze=False,
            encoder_weights=None
)
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
model.fit(train_gen, validation_data=val_gen,steps_per_epoch=80,epochs=3)


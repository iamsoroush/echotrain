# requirements

from dataset_generator import DatasetGenerator
from dataset_base import DatasetBase
from glob import glob  # for listing the directory of dataset
import random
import configparser
import numpy as np
import pandas as pd
import os


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
        age: patients between two ages in list, list
        sex: sex of patient, can be female(F) or male(M), list
        stage: stage of heart in image, can be end_systolic(ES) or end_dyastolic(ED), list
        view: view of the hear image, can be two chamber view(2CH) or four chamber view(4CH), list
        image_quality: quality of image in dataset, can be 'Good', 'Medium', 'Poor', list
        """

        self.age = config.data_handler.dataset_features.age
        self.sex = config.data_handler.dataset_features.sex
        self.stage = config.data_handler.dataset_features.stage
        self.view = config.data_handler.dataset_features.view
        self.image_quality = config.data_handler.dataset_features.image_quality

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
        self._build_data_frame()

    def create_data_generators(self):

        """Creates data generators based on batch_size, input_size

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """
        # dataset_dir = self.dataset_dir

        list_images_dir, list_labels_dir = self._fetch_data()

        # shuffling
        if self.shuffle:
            list_images_dir, list_labels_dir = self._shuffle_func(list_images_dir,
                                                                  list_labels_dir)
        # splitting
        x_train_dir, y_train_dir, x_val_dir, y_val_dir = self._split(list_images_dir,
                                                                     list_labels_dir,
                                                                     self.split_ratio)

        # adding 'train' and 'validation' status to the data-frame
        self.add_train_val_to_data_frame(x_train_dir, x_val_dir)

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

        list_images_dir, list_labels_dir = self._fetch_data()

        dataset_gen = DatasetGenerator(list_images_dir, list_labels_dir, self.batch_size,
                                       self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)

        n_iter_dataset = dataset_gen.get_n_iter()

        return dataset_gen, n_iter_dataset

    def get_data_frame(self):
        """

        :return pandas.DataFrame of all features of each data in dataset
        """
        return self.df_dataset

    def _fetch_data(self):

        """
        fetches data from directory of A4C view images of CAMUS dataset

        dataset_dir: directory address of the dataset

        :return list_images_dir: list of the A4C view images directory
        :return list_labels_dir: list of the type_map labels directory
        """

        data_dir = self.df_dataset[self.df_dataset['view'].isin(self.view) &
                                   self.df_dataset['stage'].isin(self.stage) &
                                   self.df_dataset['image_quality'].isin(self.image_quality) &
                                   self.df_dataset['sex'].isin(self.sex) &
                                   (self.df_dataset['age'] >= self.age[0]) &
                                   (self.df_dataset['age'] <= self.age[1])][['patient_id',
                                                                             'mhd_image_filename',
                                                                             'mhd_label_filename']]

        x_dir = np.array([os.path.join(self.dataset_dir, patient_id, patient_image_dir)
                          for patient_id, patient_image_dir in zip(data_dir['patient_id'],
                                                                   data_dir['mhd_image_filename'])])

        y_dir = np.array([os.path.join(self.dataset_dir, patient_id, patient_label_dir)
                          for patient_id, patient_label_dir in zip(data_dir['patient_id'],
                                                                   data_dir['mhd_label_filename'])])

        list_images_dir = x_dir
        list_labels_dir = {}
        for i in range(len(y_dir)):
            list_labels_dir[x_dir[i]] = y_dir[i]

        return list_images_dir, list_labels_dir

    def _build_data_frame(self):

        """
        This method gives you a table showing all features of each data in Pandas DataFrame format.
        Columns of this DataFrame are:
          patients: The specific number of a patient in dataset
          position: CAMUS dataset consist of 2 chamber (2CH) and 4 chamber (4CH) images
          stage: CAMUS dataset consist of end_systolic (ES), end_diastolic (ED) and sequence(between ED and ES ) data
          mhd_filename: File name of the .mhd format image
          raw_filename: File name of the .raw format image
          mhd_label_name: File name of the .mhd format labeled image
          raw_label_name: File name of the .mhd format labeled image
          ED_frame: The number of frame in sequence data that is showing ED
          ES_frame: The number of frame in sequence data that is showing ES
          NbFrame: The number of frames in sequence data
          sex: sex of the patient that can be female(F) or male(M)
          age: age of the patient
          ImageQuality: there are 3 types of image quality (Good, Medium, Poor)
          lv_edv: Left ventricle end_diastolic volume
          lv_esv: Left ventricle end_systolic volume
          lv_ef: Left ventricle ejection fraction
          status: showing if the patient is for train set or for validation

          :return Pandas DataFrame consisting features of each data in dataset
        """

        patient_dir_list = glob(os.path.join(self.dataset_dir, "*"))
        patient_dir_list.sort()

        df = {'patient_id': [],
              'mhd_image_filename': [],
              'raw_image_filename': [],
              'mhd_label_filename': [],
              'raw_label_filename': [],
              'view': [],
              'stage': [],
              'sex': [],
              'age': [],
              'ed_frame': [],
              'es_frame': [],
              'image_quality': [],
              'lv_edv': [],
              'lv_esv': [],
              'lv_ef': [],
              'num_of_frame': [],
              'status': []}

        config_parser = configparser.ConfigParser()

        for patient_dir in patient_dir_list[:450]:
            info_2ch = open(os.path.join(patient_dir, "Info_2CH.cfg"))
            info_4ch = open(os.path.join(patient_dir, "Info_4CH.cfg"))
            echo_data_list = [str(p_dir.split('.')[0]) for p_dir in os.listdir(patient_dir)
                              if p_dir.split('.')[-1] == 'mhd' and "gt" not in p_dir.split('_')[-1]]

            for echo_data in echo_data_list:
                elements = echo_data.split('_')
                df['patient_id'].append(elements[0])
                df['view'].append(elements[1])
                df['stage'].append(elements[2])

                df['mhd_image_filename'].append(f'{elements[0]}_{elements[1]}_{elements[2]}.mhd')
                df['raw_image_filename'].append(f'{elements[0]}_{elements[1]}_{elements[2]}.raw')
                if elements[2] != 'sequence':
                    df['mhd_label_filename'].append(f'{elements[0]}_{elements[1]}_{elements[2]}_gt.mhd')
                    df['raw_label_filename'].append(f'{elements[0]}_{elements[1]}_{elements[2]}_gt.raw')
                else:
                    df['mhd_label_filename'].append('None')
                    df['raw_label_filename'].append('None')

                if elements[1] == '2CH':
                    config_file = info_2ch
                else:
                    config_file = info_4ch

                config_parser.read_string(f'[{elements[1]}]\n' + config_file.read())
                df['ed_frame'].append(int(config_parser.get(elements[1], 'ED')))
                df['es_frame'].append(int(config_parser.get(elements[1], 'ES')))
                df['num_of_frame'].append(int(config_parser.get(elements[1], 'NbFrame')))
                df['sex'].append(config_parser.get(elements[1], 'Sex'))
                df['age'].append(float(config_parser.get(elements[1], 'Age')))
                df['image_quality'].append(config_parser.get(elements[1], 'ImageQuality'))
                df['lv_edv'].append(float(config_parser.get(elements[1], 'LVedv')))
                df['lv_esv'].append(float(config_parser.get(elements[1], 'LVesv')))
                df['lv_ef'].append(float(config_parser.get(elements[1], 'LVef')))
                df['status'].append('train')

        self.df_dataset = pd.DataFrame(df)

    def add_train_val_to_data_frame(self, train_dir, val_dir):
        """
        adding the updates of the status of the patients

        :param train_dir: train set directory
        :param val_dir: validation set directory
        """
        for each_dir in train_dir:
            patient_id = each_dir.replace('\\', '/').split('/')[-2]
            self.df_dataset.loc[self.df_dataset['patient_id'] == patient_id, 'status'] = 'train'

        for each_dir in val_dir:
            patient_id = each_dir.replace('\\', '/').split('/')[-2]
            self.df_dataset.loc[self.df_dataset['patient_id'] == patient_id, 'status'] = 'validation'

    def _shuffle_func(self, x, y):

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
    def _split(x, y, split_ratio):

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


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
    and by calling the ``create_data_generators`` or ``create_test_data_generator`` function,
    reads the data from the given directory as follow:

    Example::

        dataset = CAMUSDataset(config.data_handler)

        # for training set:
        train_gen, val_gen, n_iter_train, n_iter_val= dataset.create_data_generators(dataset_dir)

        # for test set:
        dataset_gen = dataset.create_test_generator(test_set_dir)

    Attributes:

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

    def __init__(self, config=None):

        """
        if config==None, default values will be invoked using self._set_efault_values

        """

        super(CAMUSDataset, self).__init__(config)

        self._load_config(config)

        self.df_dataset = None
        self._build_data_frame()

        self.list_images_dir, self.list_labels_dir = self._fetch_data()
        if self.shuffle:
            self.list_images_dir, self.list_labels_dir = self._shuffle_func(self.list_images_dir,
                                                                            self.list_labels_dir)
        # splitting
        train_indices, val_indices = self._split_indexes(self._clean_data_df.index)

        self.train_df_ = self._clean_data_df.loc[train_indices]
        self.val_df_ = self._clean_data_df.loc[val_indices]

        self.x_train_dir = np.array(self.train_df_['image_path'].to_list())
        self.y_train_dir = np.array(self.train_df_['label_path'].to_list())
        self.y_train_dir = dict(zip(self.x_train_dir, self.y_train_dir))

        self.x_val_dir = np.array(self.val_df_['image_path'].to_list())
        self.y_val_dir = np.array(self.val_df_['label_path'].to_list())
        self.y_val_dir = dict(zip(self.x_val_dir, self.y_val_dir))

        # self.x_train_dir, self.y_train_dir, self.x_val_dir, self.y_val_dir = self._split(self.list_images_dir,
        #                                                                                  self.list_labels_dir,
        #                                                                                  self.split_ratio)

        # adding 'train' and 'validation' status to the data-frame
        self._add_train_val_to_data_frame(self.x_train_dir, self.x_val_dir)

    def create_train_data_generator(self):

        """Train data generator"""

        train_data_gen = DatasetGenerator(self.x_train_dir,
                                          self.y_train_dir,
                                          self.batch_size,
                                          self.input_size,
                                          self.n_channels,
                                          self.to_fit,
                                          self.shuffle,
                                          self.seed)
        n_iter_train = train_data_gen.get_n_iter()
        return train_data_gen, n_iter_train

    def create_validation_data_generator(self):

        """Validation data generator

        Here we will set ``shuffle=False`` because we don't need shuffling for validation data.
        """

        val_data_gen = DatasetGenerator(self.x_val_dir,
                                        self.y_val_dir,
                                        self.batch_size,
                                        self.input_size,
                                        self.n_channels,
                                        self.to_fit,
                                        shuffle=False)
        n_iter_val = val_data_gen.get_n_iter()
        return val_data_gen, n_iter_val

    def create_test_data_generator(self):

        """
        Creates data generators based on ``batch_size``, ``input_size``

        :returns dataset_gen: training data generator which yields ``(batch_size, h, w, c)`` tensors
        :returns n_iter_dataset: number of iterations per epoch for train_data_gen
        """

        list_images_dir, list_labels_dir = self._fetch_data()

        dataset_gen = DatasetGenerator(list_images_dir, list_labels_dir, self.batch_size,
                                       self.input_size, self.n_channels, self.to_fit, self.shuffle, self.seed)

        n_iter_dataset = dataset_gen.get_n_iter()

        return dataset_gen, n_iter_dataset

    @property
    def raw_df(self):

        """

        :return pandas.DataFrame of all features of each data in dataset
        """

        return self.df_dataset

    @property
    def train_df(self):
        return self.train_df_

    @property
    def validation_df(self):
        return self.val_df_

    def _load_config(self, config):

        """Load all parameters from config file"""

        self._set_default_params()

        if config is not None:
            cfg_dh = config.data_handler
            self.age = cfg_dh.dataset_features.age
            self.sex = cfg_dh.dataset_features.sex
            self.stage = cfg_dh.dataset_features.stage
            self.view = cfg_dh.dataset_features.view
            self.image_quality = cfg_dh.dataset_features.image_quality
            self.batch_size = cfg_dh.batch_size
            self.input_h = config.input_h
            self.input_w = config.input_w
            # self.input_size = (self.input_h, self.input_w)
            self.n_channels = config.n_channels
            self.split_ratio = cfg_dh.split_ratio
            self.seed = cfg_dh.seed
            self.shuffle = cfg_dh.shuffle
            self.to_fit = cfg_dh.to_fit
            self.dataset_dir = cfg_dh.dataset_dir

    @property
    def input_size(self):
        return self.input_h, self.input_w

    def _set_default_params(self):

        """Default values for parameters"""

        self.age = [10, 100]
        self.sex = ['M', 'F']
        self.stage = ['ED', 'ES']
        self.view = ["4CH"]
        self.image_quality = ["Poor", "Medium", "Good"]

        self.batch_size = 8
        self.input_h = 256
        self.input_w = 256
        # self.input_size = (self.input_h, self.input_w)
        self.n_channels = 1
        self.split_ratio = 0.8
        self.seed = 101
        self.shuffle = True
        self.to_fit = True
        self.dataset_dir = 'training'

    def _fetch_data(self):

        """
        fetches data from directory of A4C view images of CAMUS dataset

        dataset_dir: directory address of the dataset

        :return list_images_dir: list of the A4C view image paths
        :return dict_labels_dir: dictionary of the type_map label paths
        """

        self._clean_data_df = self.df_dataset[self.df_dataset['view'].isin(self.view) &
                                              self.df_dataset['stage'].isin(self.stage) &
                                              self.df_dataset['image_quality'].isin(self.image_quality) &
                                              self.df_dataset['sex'].isin(self.sex) &
                                              (self.df_dataset['age'] >= self.age[0]) &
                                              (self.df_dataset['age'] <= self.age[1])]

        self._clean_data_df['image_path'] = self._clean_data_df.apply(
            lambda x: os.path.join(self.dataset_dir, x['patient_id'], x['mhd_image_filename']), axis=1)
        self._clean_data_df['label_path'] = self._clean_data_df.apply(
            lambda x: os.path.join(self.dataset_dir, x['patient_id'], x['mhd_label_filename']), axis=1)

        # data_dir = self._clean_data_df[['patient_id',
        #                                 'mhd_image_filename',
        #                                 'mhd_label_filename']]
        #
        # x_dir = np.array([os.path.join(self.dataset_dir, patient_id, patient_image_dir)
        #                   for patient_id, patient_image_dir in zip(data_dir['patient_id'],
        #                                                            data_dir['mhd_image_filename'])])
        #
        # y_dir = np.array([os.path.join(self.dataset_dir, patient_id, patient_label_dir)
        #                   for patient_id, patient_label_dir in zip(data_dir['patient_id'],
        #                                                            data_dir['mhd_label_filename'])])

        x_dir = list(self._clean_data_df['image_path'].unique())
        y_dir = list(self._clean_data_df['label_path'].unique())

        list_images_dir = x_dir
        dict_labels_dir = {}
        for i in range(len(y_dir)):
            dict_labels_dir[x_dir[i]] = y_dir[i]

        return list_images_dir, dict_labels_dir

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

    def _add_train_val_to_data_frame(self, train_dir, val_dir):

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

    # @staticmethod
    # def _split(x, y, split_ratio):
    #
    #     """
    #     splits the dataset into train and validation set by the corresponding ratio
    #     the ratio is "train portion/whole data"
    #
    #     :param x: list of images, np.ndarray
    #     :param y: list of segmentation labels, np.ndarray
    #     :param split_ratio: split ratio for trainset, float
    #
    #     :return x_train: images train_set, np.ndarray
    #     :return y_train: segmentation labels train_set, np.ndarray
    #     :return x_val: images validation_set, np.ndarray
    #     :return y_val: segmentation labels validation_set, np.ndarray
    #     """
    #
    #     # set train size by split_ratio var
    #     train_size = round(len(x) * split_ratio)
    #
    #     # splitting
    #     x_train = x[:train_size]
    #     y_train = dict(list(y.items())[:train_size])
    #
    #     x_val = x[train_size:]
    #     y_val = dict(list(y.items())[train_size:])
    #
    #     return x_train, y_train, x_val, y_val

    def _split_indexes(self, indexes):
        train_size = round(len(indexes) * self.split_ratio)
        train_indices = indexes[:train_size]
        val_indices = indexes[train_size:]
        return train_indices, val_indices


if __name__ == '__main__':
    # from dataset_echonet import EchoNetDataset
    from echotrain.model.pre_processing import PreProcessor
    from echotrain.utils.handling_yaml import load_config_file
    import matplotlib.pyplot as plt

    config_path = "../config/config_example.yaml"
    config = load_config_file(config_path)
    dataset = CAMUSDataset(config)

    train_gen, val_gen, n_iter_train, n_iter_val = dataset.create_data_generators()
    test_gen, n_iter_test = dataset.create_test_data_generator()

    print(n_iter_train)
    preprocessor = PreProcessor(config)
    preprocessed_train_gen = preprocessor.add_preprocess(train_gen, False)

    for i in range(10):
        batch = next(preprocessed_train_gen)
        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(10, 5)
        print(batch[0][0].shape)
        ax[0].axis('off')
        ax[0].imshow(batch[0][0], cmap='gray')
        ax[1].axis('off')
        ax[1].imshow(batch[1][0], cmap='gray')
        plt.show()
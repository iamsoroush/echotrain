# requirements

from dataset_generator import DatasetGenerator
from dataset_base import DatasetBase
import random
import numpy as np
import pandas as pd
import os
from tqdm import tqdm


class EchoNetDataset(DatasetBase):
    """
    This class makes our dataset ready to use by given desired values to its parameters
    and by calling the "create_data_generators" or "create_test_data_generator" function,
    reads the data from the given directory as follow:

    HOW TO:
    dataset = EchoNetDataset(config)

    # for training set:
    train_gen, val_gen, n_iter_train, n_iter_val= dataset.create_data_generators()

    # for test set:
    test_gen = dataset.create_test_data_generator()
    """

    def __init__(self, config=None):

        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        if config==None, default values will be invoked using self._set_efault_values

        batch_size: batch size, int
        input_size: input image resolution, (h, w)
        n_channels: number of channels, int
        to_fit: for predicting time, bool
        shuffle: if True the dataset will shuffle with random_state of seed, bool
        seed: seed, int
        stage: stage of heart in image, can be end_systolic(ES) or end_dyastolic(ED), list
        view: view of the hear image, can be four chamber view (4CH), list
        df_dataset: information dataframe of the whole dataset, pd.DataFrame
        _clean_data_df: contains the desired field inforamtions with image and labels full pathes
        train_df_: information dataframe of train set, pd.DataFrame
        val_df_: information dataframe of validation set, pd.DataFrame
        test_df_: information dataframe of test set, pd.DataFrame
        """

        super(EchoNetDataset, self).__init__(config)

        self._load_config(config)

        self.df_dataset = None
        self._build_data_frame()

        self.list_images_dir, self.list_labels_dir = self._fetch_data()
        if self.shuffle:
            self.list_images_dir, self.list_labels_dir = self._shuffle_func(self.list_images_dir,
                                                                            self.list_labels_dir)
        # splitting
        self._split_indexes()

        self.train_df_ = self._clean_data_df.loc[self.train_indices]
        self.val_df_ = self._clean_data_df.loc[self.val_indices]
        self.test_df_ = self._clean_data_df.loc[self.test_indices]

        self.x_train_dir = np.array(self.train_df_['image_path'].to_list())
        self.y_train_dir = np.array(self.train_df_['label_path'].to_list())
        self.y_train_dir = dict(zip(self.x_train_dir, self.y_train_dir))

        self.x_val_dir = np.array(self.val_df_['image_path'].to_list())
        self.y_val_dir = np.array(self.val_df_['label_path'].to_list())
        self.y_val_dir = dict(zip(self.x_val_dir, self.y_val_dir))

        self.x_test_dir = np.array(self.test_df_['image_path'].to_list())
        self.y_test_dir = np.array(self.test_df_['label_path'].to_list())
        self.y_test_dir = dict(zip(self.x_test_dir, self.y_test_dir))

        # self.x_train_dir, self.y_train_dir, self.x_val_dir, self.y_val_dir = self._split(self.list_images_dir,
        #                                                                                  self.list_labels_dir,
        #                                                                                  self.split_ratio)

        # # adding 'train' and 'validation' status to the data-frame
        # self._add_train_val_to_data_frame(self.x_train_dir, self.x_val_dir)

    def create_data_generators(self):

        """Creates data generators based on batch_size, input_size

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """

        train_data_gen, n_iter_train = self.create_train_data_generator()
        val_data_gen, n_iter_val = self.create_validation_data_generator()

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

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

        Here we will set shuffle=False because we don't need shuffling for validation data.
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
        Creates data generators based on batch_size, input_size

        :returns dataset_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_dataset: number of iterations per epoch for train_data_gen
        """

        test_data_gen = DatasetGenerator(self.x_test_dir,
                                         self.y_test_dir,
                                         self.batch_size,
                                         self.input_size,
                                         self.n_channels,
                                         self.to_fit,
                                         shuffle=False)
        n_iter_test = test_data_gen.get_n_iter()
        return test_data_gen, n_iter_test

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

    @property
    def test_df(self):
        return self.test_df_

    def _load_config(self, config):

        """Load all parameters from config file"""

        self._set_default_params()

        if config is not None:
            cfg_dh = config.data_handler
            self.stage = cfg_dh.dataset_features.stage
            self.view = cfg_dh.dataset_features.view
            self.batch_size = cfg_dh.batch_size
            self.input_h = config.input_h
            self.input_w = config.input_w
            # self.input_size = (self.input_h, self.input_w)
            self.n_channels = config.n_channels
            # self.split_ratio = cfg_dh.split_ratio
            self.seed = cfg_dh.seed
            self.shuffle = cfg_dh.shuffle
            self.to_fit = cfg_dh.to_fit
            self.dataset_dir = cfg_dh.dataset_dir
            self.info_df_dir = os.path.join(self.dataset_dir, 'info_df.csv')

    @property
    def input_size(self):
        return self.input_h, self.input_w

    def _set_default_params(self):

        """Default values for parameters"""

        self.stage = ['ED', 'ES']
        self.view = ["4CH"]

        self.batch_size = 8
        self.input_h = 256
        self.input_w = 256
        # self.input_size = (self.input_h, self.input_w)
        self.n_channels = 1
        # self.split_ratio = 0.8
        self.seed = 101
        self.shuffle = True
        self.to_fit = True
        self.dataset_dir = 'EchoNet-Dynamic'
        self.info_df_dir = os.path.join(self.dataset_dir, 'info_df.csv')

    def _fetch_data(self):

        """
        fetches data from directory of A4C view images of EchoNet-Dynamic dataset

        dataset_dir: directory address of the dataset

        :return list_images_dir: list of the desired images view directories
        :return dict_labels_dir: dictionary of the type_map label paths
        """

        self._clean_data_df = self.df_dataset[self.df_dataset['view'].isin(self.view) &
                                              self.df_dataset['stage'].isin(self.stage)]

        self._clean_data_df['image_path'] = self._clean_data_df.apply(
            lambda x: os.path.join(self.dataset_dir, 'Cases/', x['case_id'], x['mhd_image_filename']), axis=1)

        self._clean_data_df['label_path'] = self._clean_data_df.apply(
            lambda x: os.path.join(self.dataset_dir, 'Cases/', x['case_id'], x['mhd_label_filename']), axis=1)

        # data_dir = self._clean_data_df[['case_id',
        #                                 'mhd_image_filename',
        #                                 'mhd_label_filename']]

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
          cases: The specific number of a case in dataset
          position: EchoNet-Dynamic dataset consists of 4 chamber (4CH) images
          stage: new EchoNet-Dynamic dataset consists of end_systolic (ES), end_diastolic (ED)
          mhd_filename: File name of the .mhd format image
          raw_filename: File name of the .raw format image
          mhd_label_name: File name of the .mhd format labeled image
          raw_label_name: File name of the .mhd format labeled image
          ED_frame: The number of frame in corresponding video that is showing ED
          ES_frame: The number of frame in corresponding video data that is showing ES
          NbFrame: The number of frames in corresponding video
          lv_edv: Left ventricle end_diastolic volume
          lv_esv: Left ventricle end_systolic volume
          lv_ef: Left ventricle ejection fraction
          status: showing weather the case is for train, validation or test set

          df_dataset: Pandas DataFrame consisting features of each data in dataset
        """

        # checking if the dataframe already existed or not
        if os.path.exists(self.info_df_dir):
            self.df_dataset = pd.read_csv(self.info_df_dir)
        else:
            file_list_df = pd.read_csv(os.path.join(self.dataset_dir, "FileList.csv"))
            volume_tracing_df = pd.read_csv(os.path.join(self.dataset_dir, 'VolumeTracings.csv'))

            stages = ['ES', 'ED']

            df = {'case_id': [],
                  'mhd_image_filename': [],
                  'raw_image_filename': [],
                  'mhd_label_filename': [],
                  'raw_label_filename': [],
                  'video_file_dir': [],
                  'view': [],
                  'stage': [],
                  'ed_frame': [],
                  'es_frame': [],
                  'lv_edv': [],
                  'lv_esv': [],
                  'lv_ef': [],
                  'num_of_frame': [],
                  'fps': [],
                  'status': []}

            # finding the miss matches between the file_list and volume_tracing dataset
            # there are 6 videos which doesn't have label so we have to ignore them
            vt_filename_unique = np.array(list(map(lambda x: x.split('.')[0], volume_tracing_df['FileName'].unique())))
            fl_filename_unique = file_list_df['FileName'].unique()
            data_diff = np.setdiff1d(fl_filename_unique, vt_filename_unique)

            for case in tqdm(file_list_df['FileName'][:]):
                if case in data_diff:
                    continue
                case_file_list = file_list_df[file_list_df['FileName'] == case]
                case_volume_tracing = volume_tracing_df[volume_tracing_df['FileName'] == f'{case}.avi']
                ED_ES_num_frames = case_volume_tracing['Frame'].unique()

                for stage in stages:
                    df['case_id'].append(case)
                    df['mhd_image_filename'].append(f'{case}_{stage}.mhd')
                    df['raw_image_filename'].append(f'{case}_{stage}.raw')
                    df['mhd_label_filename'].append(f'{case}_{stage}_gt.mhd')
                    df['raw_label_filename'].append(f'{case}_{stage}_gt.raw')
                    df['video_file_dir'].append(f'Videos/{case}.avi')
                    df['view'].append('4CH')
                    df['stage'].append(stage)
                    df['ed_frame'].append(ED_ES_num_frames[0])
                    df['es_frame'].append(ED_ES_num_frames[1])
                    df['lv_edv'].append(float(case_file_list['EDV']))
                    df['lv_esv'].append(float(case_file_list['ESV']))
                    df['lv_ef'].append(float(case_file_list['EF']))
                    df['num_of_frame'].append(int(case_file_list['NumberOfFrames']))
                    df['fps'].append(int(case_file_list['FPS']))
                    df['status'].append(str(case_file_list['Split'].values[0]))

            self.df_dataset = pd.DataFrame(df)
            self.df_dataset.to_csv(self.info_df_dir, index=False)

    # def _add_train_val_to_data_frame(self, train_dir, val_dir):
    #
    #     """
    #     adding the updates of the status of the patients
    #
    #     :param train_dir: train set directory
    #     :param val_dir: validation set directory
    #     """
    #
    #     for each_dir in train_dir:
    #         case_id = each_dir.replace('\\', '/').split('/')[-2]
    #         self.df_dataset.loc[self.df_dataset['case_id'] == case_id, 'status'] = 'train'
    #
    #     for each_dir in val_dir:
    #         case_id = each_dir.replace('\\', '/').split('/')[-2]
    #         self.df_dataset.loc[self.df_dataset['case_id'] == case_id, 'status'] = 'validation'

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

    def _split_indexes(self):
        """
        making splitting indexes for train, validation, test set from the echonet dataframe
        """
        self.indexes = self._clean_data_df.index
        self.train_indices = self.indexes[self._clean_data_df['status'] == 'TRAIN']
        self.val_indices = self.indexes[self._clean_data_df['status'] == 'VAL']
        self.test_indices = self.indexes[self._clean_data_df['status'] == 'TEST']

if __name__ == '__main__':
    # from dataset_echonet import EchoNetDataset
    from echotrain.model.pre_processing import PreProcessor
    from echotrain.utils.handling_yaml import load_config_file
    import matplotlib.pyplot as plt

    config_path = "../config/config_example_echonet.yaml"
    config = load_config_file(config_path)
    dataset = EchoNetDataset(config)

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
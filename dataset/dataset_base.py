from glob import glob # for listing the directory of dataset
import skimage.io as io # to read the .mhd and .raw data
import os

class DatasetBase:

    def __init__(self, batch_size, input_res, ):

        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        :param batch_size: batch size, int
        :param input_res: input image resolution, (h, w, c)
        """

        self.batch_size = batch_size
        self.input_res = input_res

    def create_data_generators(self, dataset_dir, ):

        """Creates data generators based on batch_size, input_res

        :param dataset_dir: dataset's directory

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """
    def fetch_data(self, dataset_dir="D:/AIMedic/FinalProject_echocardiogram/echoC_Dataset/CAMUS/training/"):

        """
        fetching data from directory of A4C view images of CAMUS dataset

        :param dataset_dir: directory address of the dataset

        :return: data_dir_list: list of the directory of each A4C view image
        :return: labels: list of the directory of each type map labels
        """
        X_4CH_ED_dir = glob(os.path.join(dataset_dir,'/*/*_4CH_ED.mhd'))
        y_4CH_ED_dir = glob(os.path.join(dataset_dir,'/*/*_4CH_ED_gt.mhd'))

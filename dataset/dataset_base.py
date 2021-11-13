import sys
sys.path.append('../echotrain/')

from abc import abstractmethod

from model.base_class import BaseClass


class DatasetBase(BaseClass):
    """
    Handles data ingestion: preparing, pre-processing, augmentation, data generators

    """

    def create_data_generators(self):

        """Creates data generators based on ``batch_size``, ``input_res``

        :returns train_data_gen: training data generator which yields ``(batch_size, h, w, c)`` tensors
        :returns val_data_gen: validation data generator which yields ``(batch_size, h, w, c)`` tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """

        train_data_gen, n_iter_train = self.create_train_data_generator()
        val_data_gen, n_iter_val = self.create_validation_data_generator()

        return train_data_gen, val_data_gen, n_iter_train, n_iter_val

    @abstractmethod
    def create_train_data_generator(self):

        """Training data generator

        :returns train_data_gen:
        :returns n_iter_train:
        """

        pass

    @abstractmethod
    def create_validation_data_generator(self):

        """Validation data generator

        Here we have set ``shuffle=False`` because we don't need shuffling for validation data.

        :returns validation_data_gen:
        :returns n_iter_val:
        """

        pass

    @property
    @abstractmethod
    def raw_df(self):

        """A table showing all features of each data in Pandas DataFrame format.

        """

        pass

    @property
    @abstractmethod
    def train_df(self):

        """The training dataframe, each row is a training instance which will provide image/label path and
            other properties of that instance.
        """

        pass

    @property
    @abstractmethod
    def validation_df(self):

        """The validation dataframe, each row is a training instance which will provide image/label path and
            other properties of that instance.
        """

        pass

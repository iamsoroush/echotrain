class DatasetBase:

    def __init__(self, config):

        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        if config==None, default values will be invoked using self._set_efault_values

        :param config: dictionary of {config_name: config_value}
        """

        self.config = config

    def create_data_generators(self):

        """Creates data generators based on batch_size, input_res

        :param dataset_dir: dataset's directory

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """

        raise Exception('not implemented!')

    def get_data_frame(self):
        """This method gives you a table showing all features of each data in Pandas DataFrame format.

        :return pandas.DataFrame of all features of each data in dataset
        """

        raise Exception('not implemented')

class DatasetBase:

    def __init__(self, config):

        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        :param config: dictionary of {config_name: config_value}
        """

        self.config = config

    def create_data_generators(self, dataset_dir):

        """Creates data generators based on batch_size, input_res

        :param dataset_dir: dataset's directory

        :returns train_data_gen: training data generator which yields (batch_size, h, w, c) tensors
        :returns val_data_gen: validation data generator which yields (batch_size, h, w, c) tensors
        :returns n_iter_train: number of iterations per epoch for train_data_gen
        :returns n_iter_val: number of iterations per epoch for val_data_gen

        """

        raise Exception('not implemented!')

class DatasetBase:

    def __init__(self, batch_size, input_res, config):

        """
        Handles data ingestion: preparing, pre-processing, augmentation, data generators

        :param batch_size: batch size, int
        :param input_res: input image resolution, (h, w, c)
        :param config: dictionary of {config_name: config_value}
        """

        self.batch_size = batch_size
        self.input_res = input_res
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

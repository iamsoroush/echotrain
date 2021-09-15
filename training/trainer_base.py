

class TrainerBase:

    def __init__(self, checkpoints_dir, logs_dir, config):

        """
        handles: MLFlow, paths, callbacks(tensorboard, lr, model checkpointin, ...), training

        model's checkpoints => checkpoints_dir/{model.name}
        model's logs (tensorboard) => logs_dir/{model.name}

        :param checkpoints_dir: checkpoints directory
        :param logs_dir: logs directory
        :param config: dictionary of {config_name: config_value}
        """

        self.checkpoints_dir = checkpoints_dir
        self.logs_dir = logs_dir
        self.config = config

    def train(self, model, train_data_gen, val_data_gen, n_iter_train, n_iter_val):

        """Trains the model on given data generators.

        Use Dataset and Model classes fir

        :param model: tensorflow model to be trained, it has to have a `fit` method
        :param train_data_gen: training data generator
        :param val_data_gen: validation data generator
        :param n_iter_train: iterations per epoch for train_data_gen
        :param n_iter_val: iterations per epoch for val_data_gen

        """

        raise Exception('not implemented')

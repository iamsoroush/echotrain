

class TrainerBase:

    def __init__(self, base_dir, config):

        """
        handles: MLFlow, paths, callbacks(tensorboard, lr, model checkpointin, ...), continous training

        tensorboard_logs => base_dir/logs
        checkpoints => base_dir/checkpoints

        :param base_dir: experiment directory, containing config.yaml file
        :param config: a Python object with attributes as config values

        Attributes
            epochs int: number of epochs for training

        """

        self.base_dir = base_dir
        self.epochs = config.epochs
        self.callbacks_config = config.callbacks

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

    def export(self):

        """Exports the best version of the weights of the model, and config.yaml file into exported sub_directory"""

        raise Exception('not implemented')

from tensorflow import keras
import tensorflow as tf
import mlflow
import os


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
        self.checkpoints_addr = self.base_dir + '/checkpoints'
        if not os.path.isdir(self.checkpoints_addr):
            os.makedirs(self.checkpoints_addr)
        self.my_callbacks = [
            # keras.callbacks.EarlyStopping(patience=2),
            keras.callbacks.ModelCheckpoint(filepath=self.checkpoints_addr+'/model.{epoch:03d}-{val_loss:.2f}.hdf5',
                                            save_weights_only=True,
                                            save_freq=self.callbacks_config.checkpoints.save_freq),
            keras.callbacks.TensorBoard(log_dir=self.base_dir+'/logs',
                                        update_freq=self.callbacks_config.tensorboard.update_freq),
        ]

    def train(self, model, train_data_gen, val_data_gen, n_iter_train, n_iter_val):
        """Trains the model on given data generators.

        Use Dataset and Model classes fir

        :param model: tensorflow model to be trained, it has to have a `fit` method
        :param train_data_gen: training data generator
        :param val_data_gen: validation data generator
        :param n_iter_train: iterations per epoch for train_data_gen
        :param n_iter_val: iterations per epoch for val_data_gen

        """
        if os.path.isdir(self.checkpoints_addr):
            latest = tf.train.latest_checkpoint(self.checkpoints_addr)
            model.load_weights(latest)

        history = model.fit(train_data_gen, steps_per_epoch=n_iter_train, epochs=self.epochs,
                            validation_data=val_data_gen, validation_steps=n_iter_val, callbacks=self.my_callbacks)
        return history

    def export(self):
        """Exports the best version of the weights of the model, and config.yaml file into exported sub_directory"""

        raise Exception('not implemented')

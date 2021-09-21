from tensorflow import keras
from shutil import copy
import mlflow
import os
import re


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
        self.evaluation_metrics = self.callbacks_config.checkpoints.evaluation_metrics
        self.export_config = config.export
        self.checkpoints_addr = self.base_dir + '/checkpoints'
        self.tensorboard_log = self.base_dir + '/logs'
        if not os.path.isdir(self.checkpoints_addr):
            os.makedirs(self.checkpoints_addr)
            os.makedirs(self.tensorboard_log)

        met_str = ''
        for i in self.evaluation_metrics:
            met_str += '-{'
            met_str += '{metric}'.format(metric=i + ':.2f')
            met_str += '}'
        checkpoints_name = '/model.{epoch:03d}' + met_str + '.hdf5'
        print(checkpoints_name)

        self.my_callbacks = [
            # keras.callbacks.EarlyStopping(patience=2),
            keras.callbacks.ModelCheckpoint(filepath=self.checkpoints_addr + checkpoints_name,
                                            save_weights_only=True,
                                            save_freq=self.callbacks_config.checkpoints.save_freq),
            keras.callbacks.TensorBoard(log_dir=self.tensorboard_log,
                                        update_freq=self.callbacks_config.tensorboard.update_freq),
        ]

    def _train(self, model, train_data_gen, val_data_gen, n_iter_train, n_iter_val):
        """Trains the model on given data generators.

        Use Dataset and Model classes fir

        :param model: tensorflow model to be trained, it has to have a `fit` method
        :param train_data_gen: training data generator
        :param val_data_gen: validation data generator
        :param n_iter_train: iterations per epoch for train_data_gen
        :param n_iter_val: iterations per epoch for val_data_gen

        """
        initial_epoch = 0
        if len(os.listdir(self.checkpoints_addr)):
            checkpoints = [self.checkpoints_addr + '/' + p for p in os.listdir(self.checkpoints_addr)]
            latest_checkpoint = max(checkpoints, key=os.path.getctime)
            # initial_epoch = re.findall('model\\.([0-9]+)(-[+-]?[0-9]+\\.[0-9]+)*', latest_checkpoint.split('/')[-1])
            initial_epoch = int(re.findall('model\\.([0-9]+)', latest_checkpoint.split('/')[-1])[0])
            model.load_weights(latest_checkpoint)

        mlflow.set_experiment('BaseModel')
        with mlflow.start_run(run_name='BaseModel'):
            mlflow.tensorflow.autolog(every_n_iter=5, log_models=True, disable=False, exclusive=False,
                                      disable_for_unsupported_versions=False, silent=True)
            history = model.fit(train_data_gen, steps_per_epoch=n_iter_train, initial_epoch=initial_epoch,
                                epochs=self.epochs, validation_data=val_data_gen, validation_steps=n_iter_val,
                                callbacks=self.my_callbacks)
        return history

    def _export(self):
        """Exports the best version of the weights of the model, and config.yaml file into exported sub_directory"""
        metric, mode = self.export_config.metric, self.export_config.mode
        metric_number = self.evaluation_metrics.index(metric)

        checkpoints = os.listdir(self.checkpoints_addr)
        model_info = {}
        for cp in checkpoints:
            epoch = re.findall('model\\.([0-9]+)', cp)[0]
            metric_values = cp.replace('.hdf5', '').split('-')
            model_info[epoch] = metric_values[1+metric_number]
            # metric_values = re.sub('model\\.[0-9]+', '', cp).replace('.hdf5', '')
        if mode == 'min':
            selected_model = min(model_info, key=model_info.get)
        else:
            selected_model = max(model_info, key=model_info.get)
        selected_checkpoint = checkpoints[int(selected_model)-1]
        chp_addr = self.checkpoints_addr+'/'+selected_checkpoint
        dst = 'exported'
        if not os.path.isdir(dst):
            os.makedirs(dst)
        print(chp_addr)
        copy(chp_addr, dst+'/'+selected_checkpoint)
        copy('../config/config_example.yaml', dst+'/config.yaml')

import os
import re
from shutil import copy
from pathlib import Path
import yaml

from tensorflow import keras
import mlflow
from mlflow.tracking import MlflowClient


class Trainer:

    """Handles MLFlow, paths, callbacks(tensorboard, lr, model checkpointing, ...), continuous training

    tensorboard_logs -> base_dir/logs
    checkpoints -> base_dir/checkpoints

    Attributes:

        epochs: number of epochs for training
        callbacks_config: contains some information for callbacks includes:

            - checkpoints: contains some information for checkpoints callback includes:

                - save_freq: determines when checkpoints saved it could have values like "epoch" or "batch"
                - evaluation_metrics: determines metrics which may be used for finding the best model weights in
                  export function

            - tensorboard: contains information for tensorboard callback includes:

                - update_freq: determines the frequency of time that tensorboard values will be updated it could be
                  "epoch" or "batch"

        export_config: contains some information for finding the best weights for model, includes:

            - metric: one of the metrics that defined in the evaluation_metrics list
            - mode: determine whether max value of the metric is appropriate or min,
              it could be "max" or "min" based on the chosen metric

        checkpoints_addr: the path where the checkpoints is going to save
        tensorboard_log:  the path where the tensorboard logs is going to save

    """

    def __init__(self, base_dir: Path, config):

        """
        :param base_dir: experiment directory, containing config.yaml file
        :param config: a Python object with attributes as config values
        """

        self.base_dir = base_dir
        self._load_params(config)
        self._check_for_exported()

        # mlflow.set_tracking_uri(f'file:{self.mlflow_tracking_uri}')

        if not os.path.isdir(self.checkpoints_addr):
            os.makedirs(self.checkpoints_addr)
            os.makedirs(self.tensorboard_log)

        self.my_callbacks = self._get_callbacks()

    def _get_callbacks(self):

        """Creates and returns callbacks based on config file"""

        met_str = ''
        for i in self.evaluation_metrics:
            met_str += '-{'
            met_str += '{metric}'.format(metric=i + ':.2f')
            met_str += '}'
        checkpoints_name = 'model.{epoch:03d}' + met_str + '.hdf5'
        checkpoints_template = self.checkpoints_addr.joinpath(checkpoints_name)

        if self.callbacks_config.checkpoints.monitor is None:
            save_best_only = False
        else:
            save_best_only = True

        callbacks = [
            keras.callbacks.ModelCheckpoint(filepath=str(checkpoints_template),
                                            save_weights_only=False,
                                            save_freq=self.callbacks_config.checkpoints.save_freq,
                                            save_best_only=save_best_only,
                                            monitor=self.callbacks_config.checkpoints.monitor),
            keras.callbacks.TensorBoard(log_dir=self.tensorboard_log,
                                        update_freq=self.callbacks_config.tensorboard.update_freq),
        ]
        return callbacks

    def train(self, model, train_data_gen, val_data_gen, n_iter_train, n_iter_val):

        """Trains the model on given data generators.

        Use Dataset and Model classes dir

        :param model: tensorflow model to be trained, it has to have a `fit` method
        :param train_data_gen: training data generator
        :param val_data_gen: validation data generator
        :param n_iter_train: iterations per epoch for train_data_gen
        :param n_iter_val: iterations per epoch for val_data_gen

        :returns fit_history:

        """

        initial_epoch = 0
        if len(os.listdir(self.checkpoints_addr)):
            # checkpoints = [self.checkpoints_addr.joinpath(p) for p in self.checkpoints_addr.iterdir()]
            latest_checkpoint = max(self.checkpoints_addr.iterdir(), key=os.path.getctime)
            print(f'found latest checkpoint: {latest_checkpoint}')
            # initial_epoch = re.findall('model\\.([0-9]+)(-[+-]?[0-9]+\\.[0-9]+)*', latest_checkpoint.split('/')[-1])
            initial_epoch = int(re.findall('model\\.([0-9]+)', latest_checkpoint.name)[0])
            print(f'initial epoch: {initial_epoch}')
            model.load_weights(latest_checkpoint)

        run_id = self._load_existing_run()
        active_run = self._setup_mlflow(run_id)

        # if mlflow.get_experiment_by_name(self.mlflow_experiment_name) is None:
        #
        #     mlflow.create_experiment(self.mlflow_experiment_name, self.mlflow_tracking_uri)
        # else:
        #     mlflow.set_experiment(self.mlflow_experiment_name)
        # with mlflow.start_run():
        with active_run as run:
            # Add params from config file to mlflow
            self._add_config_file_to_mlflow()

            mlflow.tensorflow.autolog(every_n_iter=1,
                                      log_models=False,
                                      disable=False,
                                      exclusive=False,
                                      disable_for_unsupported_versions=False,
                                      silent=False)

            # Write run_id
            with open(self.run_id_path, 'w') as f:
                f.write(run.info.run_id)

            # Fit
            history = model.fit(train_data_gen,
                                steps_per_epoch=n_iter_train,
                                initial_epoch=initial_epoch,
                                epochs=self.epochs,
                                validation_data=val_data_gen,
                                validation_steps=n_iter_val,
                                callbacks=self.my_callbacks)
        return history

    def export(self):

        """Exports the best version of the weights of the model, and config.yaml file into exported sub_directory.

        This method will delete all checkpoints after exporting the best one

        :returns exported_dir:
        """

        metric, mode = self.export_config.metric, self.export_config.mode
        metric_number = self.evaluation_metrics.index(metric)

        checkpoints = os.listdir(self.checkpoints_addr)
        model_info = {}
        for cp in checkpoints:
            epoch = re.findall('model\\.([0-9]+)', cp)[0]
            metric_values = cp.replace('.hdf5', '').split('-')
            model_info[epoch] = metric_values[1 + metric_number]
            # metric_values = re.sub('model\\.[0-9]+', '', cp).replace('.hdf5', '')
        if mode == 'min':
            selected_model = min(model_info, key=model_info.get)
        else:
            selected_model = max(model_info, key=model_info.get)

        selected_checkpoint = checkpoints[int(selected_model) - 1]
        chp_addr = os.path.join(self.checkpoints_addr, selected_checkpoint)
        if not self.exported_dir.is_dir():
            os.makedirs(self.exported_dir)
        copy(chp_addr, self.exported_dir.joinpath(selected_checkpoint))

        copy(self.config_file_path, self.exported_dir.joinpath('config.yaml'))

        # Delete checkpoints
        for ch in checkpoints:
            os.remove(self.checkpoints_addr.joinpath(ch))

        return self.exported_dir

    def _check_for_exported(self):

        """Raises exception if exported directory exists and contains some files"""

        if self.exported_dir.is_dir():
            if any(self.exported_dir.iterdir()):
                raise Exception('exported files already exist.')

    def _load_params(self, config):

        """Reads params from config"""

        self.epochs = config.trainer.epochs
        self.callbacks_config = config.trainer.callbacks
        self.evaluation_metrics = self.callbacks_config.checkpoints.evaluation_metrics
        self.export_config = config.trainer.export

        # Paths
        self.checkpoints_addr = self.base_dir.joinpath('checkpoints')  # os.path.join(self.base_dir, 'checkpoints')
        self.tensorboard_log = self.base_dir.joinpath('logs')  # os.path.join(self.base_dir, 'logs')
        self.run_id_path = self.base_dir.joinpath('run_id.txt')  # os.path.join(self.base_dir, 'run_id.txt')
        self.exported_dir = self.base_dir.joinpath('exported')
        config_file_name = [i for i in os.listdir(self.base_dir) if i.endswith('.yaml')][0]
        self.config_file_path = self.base_dir.joinpath(config_file_name)

        # MLFlow
        self.mlflow_tracking_uri = config.trainer.mlflow.tracking_uri
        self.mlflow_experiment_name = config.trainer.mlflow.experiment_name
        # self.run_name = config.trainer.mlflow.run_name
        self.run_name = self.base_dir.name

    def _load_existing_run(self):

        """Loads run_id if exists, if not, returns None"""

        run_id = None

        if self.run_id_path.exists():
            with open(self.run_id_path, 'r') as f:
                run_id = f.readline()

        return run_id

    def _setup_mlflow(self, run_id):

        """Sets up mlflow.

        tracking_uri/
            experiment_id/
                run1
                run2
                ...
        """

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        client = MlflowClient(self.mlflow_tracking_uri)

        if run_id is not None:
            mlflow.set_experiment(self.mlflow_experiment_name)
            active_run = mlflow.start_run(run_id=run_id)
        else:
            experiment = client.get_experiment_by_name(self.mlflow_experiment_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
            else:
                experiment_id = mlflow.create_experiment(self.mlflow_experiment_name)

            active_run = mlflow.start_run(experiment_id=experiment_id, run_name=self.run_name)

        return active_run

    def _add_config_file_to_mlflow(self):

        """Adds parameters from config file to mlflow"""

        def param_extractor(dictionary):

            """Returns a list of each item formatted like 'trainer.mlflow.tracking_uri: /tracking/uri' """

            values = []
            for key, value in dictionary.items():
                if isinstance(value, dict):
                    items_list = param_extractor(value)
                    for i in items_list:
                        values.append(f'{key}.{i}')
                else:
                    values.append(f'{key}: {value}')
            return values

        with open(self.config_file_path) as file:
            data_map = yaml.safe_load(file)

        str_params = param_extractor(data_map)
        params = {}
        for item in str_params:
            name = item.split(':')[0]
            item_value = item.split(': ')[-1]
            params[name] = item_value

        mlflow.log_params(params)

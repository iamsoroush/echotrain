import keras_tuner as kt
import os
from .hypermodel_unet_baseline import HyperModel
from tensorflow import keras
import yaml
from hpo.main_tuner import MainTuner


class HPOBaseline:
    """
    This class get config file, tune the model with designated hyperparamethers, and
    export a new config file with optimized hyperparamethers.

    """

    def __init__(self, config):
        self.config = config
        self._get_parameters()

    def _get_parameters(self):
        hpo_config = self.config.hpo
        self.objective = hpo_config.objective.name
        self.direction = hpo_config.objective.direction
        self.max_trials = hpo_config.max_trials
        self.overwrite = hpo_config.overwrite
        self.directory = hpo_config.directory
        self.project_name = hpo_config.project_name
        self.epoch_tuner = hpo_config.epoch_tuner

    def generate_tuner(self):
        """

        Returns: a keras tuner with random search with designated parameters in config file.

        """
        tuner = kt.RandomSearch(
            HyperModel(self.config),
            objective=kt.Objective(self.objective, direction=self.direction),
            max_trials=self.max_trials,
            overwrite=self.overwrite,
            directory=os.path.join(self.directory, self.project_name),
            project_name=self.project_name)

        return tuner

    def search_hp(self, train_generator, validation_generator, n_iter_train, n_iter_val, searching_type='model'):
        """

        Args:
            train_generator: train data generator
            validation_generator: validation data generator
            n_iter_train: number of iter of train data generator
            n_iter_val: number of iter of validation data generator

        Returns:a tuner with optimized hyperparamethers.

        """
        if searching_type == 'model':
            tuner = self.generate_tuner()
        elif searching_type == 'preprocessing':
            tuner = self.generate_tuner_for_preprocessing()

        tuner.search(train_generator,
                     steps_per_epoch=n_iter_train,
                     epochs=self.epoch_tuner,
                     validation_data=validation_generator,
                     validation_steps=n_iter_val,
                     callbacks=[keras.callbacks.TensorBoard(os.path.join(self.directory, 'logs'))],
                     )
        return tuner

    @staticmethod
    def get_best_hp(tuner):
        """

        Args:
            tuner: keras tuner

        Returns: best hyperparameters of searched tuner.

        """
        return tuner.get_best_hyperparameters()[0]

    @staticmethod
    def get_tuner_summary(tuner):
        """

        Args:
            tuner: keras tuner

        Returns: summary of tuner.

        """
        return tuner.results_summary()

    def export_config(self, main_config_bath, tuner):
        """
        Create config file with optimized hyperparamethers

        Args:
            main_config_bath: the first config file directory
            tuner: searched tuner

        """
        result_file = os.path.join(self.directory, 'best_hp_config.yaml')
        best_hp = tuner.get_best_hyperparameters()[0]
        best_hp_values = best_hp.values

        with open(main_config_bath, 'r') as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile)

        for key, value in cur_yaml.items():
            if key == 'model':
                for key2, value2 in value.items():
                    if key2 == 'optimizer':
                        cur_yaml['model']['optimizer']['type'] = best_hp_values['optimizer_type']
                        cur_yaml['model']['optimizer']['initial_lr'] = best_hp_values['lr']
                    elif key2 in best_hp_values.keys():
                        cur_yaml['model'][key2] = best_hp_values[key2]

        with open(result_file, 'w') as yamlfile:
            yaml.safe_dump(cur_yaml, yamlfile, default_flow_style=False)

    def generate_tuner_for_preprocessing(self):

        tuner = MainTuner(oracle=kt.oracles.BayesianOptimization(objective=kt.Objective("loss", "min")
                                                                 , max_trials=2),
                          hypermodel=HyperModel,
                          directory="results",
                          project_name="mnist_custom_training")
        return tuner

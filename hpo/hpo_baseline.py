import keras_tuner as kt
import os
from .hypermodel_unet_baseline import HyperModel
from tensorflow import keras


class HPOBaseline:

    def __init__(self, config):
        self.config = config
        self._get_parameters()

    def _get_parameters(self):
        hpo_config = self.config.hpo
        self.objective = hpo_config.objective
        self.direction = hpo_config.direction
        self.max_trials = hpo_config.max_trials
        self.overwrite = hpo_config.overwrite
        self.directory = hpo_config.directory
        self.project_name = hpo_config.project_name
        self.epoch_tuner = hpo_config.epoch_tuner

    def generate_tuner(self):
        tuner = kt.RandomSearch(
            HyperModel(self.config),
            objective=kt.Objective(self.objective, direction=self.direction),
            max_trials=self.max_trials,
            overwrite=self.overwrite,
            directory=self.directory,
            project_name=self.project_name,
        )

        return tuner

    def search_hp(self, train_generator, validation_generator, n_iter_train, n_iter_val):
        tuner = self.generate_tuner()
        # tuner.search(x, y, epochs=self.epoch_tuner)
        tuner.search(train_generator,
                     steps_per_epoch=n_iter_train,
                     epochs=self.epoch_tuner,
                     validation_data=validation_generator,
                     validation_steps=n_iter_val,
                     callbacks=[keras.callbacks.TensorBoard(self.directory + '/logs')],
                     )
        return tuner

    def get_best_hp(self, tuner):
        best_model = tuner.get_best_models()[0]
        best_model.save(os.path.join(self.directory, "model_with_best_hp_graph.h5"))
        return tuner.get_best_hyperparameters()[0]

    @staticmethod
    def get_tuner_summary(tuner):
        return tuner.result_summary()

    def retrain_model(self, train_generator, validation_generator, n_iter_train, n_iter_val):
        hypermodel = HyperModel(self.config)
        tuner = self.search_hp(train_generator, validation_generator, n_iter_train, n_iter_val)
        best_hp = self.get_best_hp(tuner)
        model = hypermodel.build(best_hp)
        # hypermodel.fit(best_hp, model, train_generator, validation_generator, epochs=1)
        hypermodel.fit(best_hp, model, train_generator, steps_per_epoch=n_iter_train,
                       validation_data=validation_generator, validation_steps=n_iter_val, epochs=1)
        model.save_weights(os.path.join(self.directory, "model_with_best_hp_weights.hdf5"))

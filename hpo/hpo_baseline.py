import keras_tuner as kt
import os
from .hypermodel_unet_baseline import HyperModel


class HPOBaseline:

    def __init__(self, config):
        self.config = config
        self._get_parameters()

    def _get_parameters(self):
        self.objective = kt.Objective("iou_coef", direction="max")
        self.max_trials = 3
        self.overwrite = True
        self.directory = ''
        self.project_name = "tune_hypermodel"
        self.epoch_tuner = 2

    def generate_tuner(self):
        tuner = kt.RandomSearch(
            HyperModel(self.config),
            objective=self.objective,
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
                     validation_steps=n_iter_val)
        return tuner

    def get_best_hp(self, tuner):
        best_model = tuner.get_best_models()[0]
        best_model.save(os.path.join(self.directory, "model_with_best_hp.h5"))
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
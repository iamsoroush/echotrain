import keras_tuner as kt
from hypermodel_unet_baseline import HyperModel


class HPOBaseline:

    def __init__(self, config):
        self.config = config

    def _get_parameters(self):
        self.objective = "val_accuracy"
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

    def tune_model(self, x, y):
        tuner = self.generate_tuner()
        tuner.search(x, y, epochs=self.epoch_tuner)
        return tuner

    @staticmethod
    def get_best_hp(tuner):
        return tuner.get_best_hyperparamethers()[0]

    def train(self, x, y):
        hypermodel = HyperModel(self.config)
        tuner = self.tune_model(x, y)
        best_hp = self.get_best_hp(tuner)

        model = hypermodel.build(best_hp)
        hypermodel.fit(best_hp, model, x, y, epochs=1)

import keras_tuner as kt
from model.baseline_unet import UNetBaseline


class HPOBaseline(kt.HyperModel):

    def __init__(self, config):
        super(HPOBaseline, self).__init__()
        self.config = config

    def build(self, hp):
        hp = kt.HyperParameters()
        unet_baseline = UNetBaseline(self.config, hp)
        model = unet_baseline.generate_training_model()

        return model

    def fit(self, model, hp, *args, **kwargs):
        return model.fit(*args, epochs=hp.Int("epochs", min_value=5, max_value=25, step=5), **kwargs)

    def _get_parameters(self):
        self.objective = "val_accuracy"
        self.max_trials = 3
        self.overwrite = True
        self.directory = ''
        self.project_name = "tune_hypermodel"
        self.epoch_tuner = 2

    def generate_tuner(self):
        tuner = kt.RandomSearch(
            HPOBaseline(self.config),
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

    def train(self, x, y):
        tuner = self.tune_model(x, y)
        best_hp = self.get_best_hp(tuner)
        model = self.build(best_hp)
        self.fit(model, best_hp, x, y)

    @staticmethod
    def get_best_hp(tuner):
        return tuner.get_best_hyperparamethers()[0]

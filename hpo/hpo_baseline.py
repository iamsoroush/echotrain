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

    def fit(self, hp, model, x, y, **kwargs):

        return model.fit(x, y, **kwargs)

    def _get_parameters(self):
        self.objecte = "val_accuracy"
        self.max_trials = 3
        self.overwrite = True
        self.directory = ''
        self.project_name = "tune_hypermodel"
        self.epoch_tuner = 2

    def generate_tuner(self):
        tuner = kt.RandomSearch(
            HPOBaseline(self.config),
            objective=self.objecte,
            max_trials=self.max_trials,
            overwrite=self.overwrite,
            directory=self.directory,
            project_name=self.project_name,
        )

        return tuner

    def tune_model(self):
        tuner = self.generate_tuner()
        tuner.search(x,y,epochs = self.epoch_tuner)
        return tuner

    @staticmethod
    def get_best_hp(tuner):
        return tuner.get_best_hyperparamethers()[0]

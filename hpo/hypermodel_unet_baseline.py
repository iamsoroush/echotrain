import keras_tuner as kt
from model.baseline_unet import UNetBaseline


class HyperModel(kt.HyperModel):

    def __init__(self, config):
        super(HyperModel, self).__init__()
        self.config = config

    def build(self, hp):
        """


        This method generate compiled model and detect hyperparameters.
        Args:
            hp: hyperparameter variables.

        Returns: model with hyperparametes
        """

        unet_baseline = UNetBaseline(self.config, hp)
        model = unet_baseline.generate_training_model()
        return model

    def fit(self, hp, model, *args, **kwargs):
        """
         Args:
             hp: hyperparameter variables.
             model: model with hyperparametes
             *args:
             **kwargs:

         Returns: trained model

         """
        return model.fit(*args, **kwargs)

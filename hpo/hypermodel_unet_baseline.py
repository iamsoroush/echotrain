from abc import ABC

import keras_tuner as kt
from model.baseline_unet import UNetBaseline


class HyperModel(kt.HyperModel, ABC):

    def __init__(self, config):
        super(HyperModel, self).__init__()
        self.config = config

    def build(self, hp):
        unet_baseline = UNetBaseline(self.config, hp)
        model = unet_baseline.generate_training_model()
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            **kwargs,
        )

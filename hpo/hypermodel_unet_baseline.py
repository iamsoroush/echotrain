import keras_tuner as kt
from model.baseline_unet import UNetBaseline
from model.pre_processing import PreProcessor


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
        train_data_gen = kwargs.get("train_data_gen")
        val_data_gen = kwargs.get("val_data_gen")

        # n_iter_val = kwargs.get("n_iter_val")
        # n_iter_train = kwargs.get("n_iter_train")

        preprocessor = PreProcessor(self.config, hp)
        train_data_gen = preprocessor.add_preprocess(train_data_gen, True)
        val_data_gen = preprocessor.add_preprocess(val_data_gen, True)

        return model.fit(train_data_gen, val_data_gen, *args, **kwargs)

import keras_tuner as kt
from model.augmentation import Augmentation
from model.pre_processing import PreProcessor
from model.baseline_unet import UNetBaseline


class PreprocessingTuner(kt.Tuner):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        augmentation = Augmentation(*args, hp=hp)
        preprocessor = PreProcessor(*args, hp=hp)
        model = UNetBaseline(*args, hp=hp).generate_training_model()

import keras_tuner as kt
from model.augmentation import Augmentation
from model.pre_processing import PreProcessor
from model.baseline_unet import UNetBaseline
from hypermodel_unet_baseline import HyperModel


class PreprocessingTuner(kt.Tuner):

    def run_trial(self, trial, *args, **kwargs):
        hp = trial.hyperparameters
        augmentation = Augmentation(*args, hp=hp)
        preprocessor = PreProcessor(*args, hp=hp)
        hyper_model = HyperModel(*args)
        model = hyper_model.build(hp)

        hyper_model.fit(train_generator,
                        steps_per_epoch=n_iter_train,
                        epochs=self.epoch_tuner,
                        validation_data=validation_generator,
                        validation_steps=n_iter_val, )

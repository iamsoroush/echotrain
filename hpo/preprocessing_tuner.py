import keras_tuner as kt
from model.augmentation import Augmentation
from model.pre_processing import PreProcessor
import tensorflow as tf
from model.baseline_unet import UNetBaseline
from .hypermodel_unet_baseline import HyperModel
import numpy as np


class PreprocessingTuner(kt.Tuner):

    def run_trial(self, trial, train_data_gen, n_iter_train, *args, **kwargs):
        hp = trial.hyperparameters
        augmentation = Augmentation(*args, hp=hp)
        preprocessor = PreProcessor(*args, hp=hp)
        model = self.hypermodel.build(trial.hyperparameters)
        train_data_gen = preprocessor.add_preprocess(train_data_gen, True)
        lr = hp.Float("learning_rate", 1e-4, 1e-2, sampling="log", default=1e-3)
        optimizer = tf.keras.optimizers.Adam(lr)
        epoch_loss_metric = tf.keras.metrics.Mean()
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True,name='loss')

        def run_train_step(data):
            # images = tf.dtypes.cast(data[0], "float32") / 255.0
            images = data[0].reshape(-1 , 128 , 128 , 1)
            labels = data[1].reshape(-1 , 128 , 128 , 1)
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            print("my loss is :" , loss)
            return loss

        for epoch in range(2):
            print("Epoch: {}".format(epoch))

            self.on_epoch_begin(trial, model, epoch, logs={})
            for batch in range(n_iter_train):
                data = next(train_data_gen)
                self.on_batch_begin(trial, model, batch, logs={})
                batch_loss = float(run_train_step(data))
                self.on_batch_end(trial, model, batch, logs={"loss": batch_loss})

                if batch % 100 == 0:
                    loss = epoch_loss_metric.result().numpy()
                    print("Batch: {}, Average Loss: {}".format(batch, loss))

            epoch_loss = epoch_loss_metric.result().numpy()
            # self.on_epoch_end(trial, model, epoch, logs={"loss": epoch_loss})
            self.oracle.update_trial(trial.trial_id, {'loss': epoch_loss})
            epoch_loss_metric.reset_states()

import keras_tuner as kt
from model.augmentation import Augmentation
from model.pre_processing import PreProcessor
import tensorflow as tf
from model.baseline_unet import UNetBaseline
from .hypermodel_unet_baseline import HyperModel
import numpy as np
from model.loss import dice_coef_loss
import tensorflow.keras as tfk


class PreprocessingTuner(kt.Tuner):

    def run_trial(self, trial, train_data_gen, n_iter_train, *args, **kwargs):
        hp = trial.hyperparameters
        unet = UNetBaseline(*args, hp=hp)
        model = unet.generate_training_model()
        preprocessor = PreProcessor(*args, hp=hp)
        train_data_gen = preprocessor.add_preprocess(train_data_gen, True)
        lr = unet.learning_rate

        optimizer_type = unet.optimizer_type
        if optimizer_type == 'adam':
            optimizer = tfk.optimizers.Adam(learning_rate=lr)
        if optimizer_type == 'sgd':
            optimizer = tfk.optimizers.SGD(learning_rate=lr)
        if optimizer_type == 'rmsprop':
            optimizer = tfk.optimizers.RMSprop(learning_rate=lr)
        if optimizer_type == 'adagrad':
            optimizer = tfk.optimizers.Adagrad(learning_rate=lr)

        loss_type = unet.loss_type
        if loss_type == 'binary_crossentropy':
            loss_fn = tf.keras.losses.BinaryCrossentropy()
        if loss_type == 'dice_coef_loss':
            loss_fn = dice_coef_loss

        epoch_loss_metric = tf.keras.metrics.Mean()

        def run_train_step(data):
            # images = tf.dtypes.cast(data[0], "float32") / 255.0
            images = data[0].reshape(-1, 128, 128, 1)
            labels = data[1].reshape(-1, 128, 128, 1)
            with tf.GradientTape() as tape:
                logits = model(images)
                loss = loss_fn(labels, logits)
                # Add any regularization losses.
                if model.losses:
                    loss += tf.math.add_n(model.losses)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss_metric.update_state(loss)
            print("my loss is :", loss)
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

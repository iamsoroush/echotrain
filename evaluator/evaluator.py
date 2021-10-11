import sys
sys.path.append('..')

from model import metric
from model import loss
import numpy as np
import pandas as pd


class Evaluator:
    """
    This class is for evaluating the model and data according to the metrics and losses

    HOW TO:
    eval = Evaluator(config)
    eval.build_data_frame(model,data_generator,n_iter)

    """

    def __init__(self, config):

        """
        :param config
        """

        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels = config.n_channels

    def build_data_frame(self, model, data_gen_val_preprocessed, n_iter, val_data_indexes):

        """Generates a report as a pandas.DataFrame

        :param val_data_indexes: index of validation data. this will be the index column of the result dataframe
        :param model: the input model which is being evaluated
        :param data_gen_val_preprocessed: the data generator which is being evaluated,
         it has to be a pre-processed data generator
        :param n_iter: number of iterations of the data generator
        :return: the dataframe which consists of the metrics and losses and also the certainty of
        the model and true positive rate and true negative rate
        """

        # building the dataframe
        new_columns = ['iou_coef_loss',
                       'dice_coef_loss',
                       'soft_dice_loss',
                       'iou_coef',
                       'sift_iou_coef',
                       'dice_coef',
                       'soft_dice',
                       'mad',
                       'hausdorff',
                       'truecertainty',
                       'falsecertainty',
                       'ambigous',
                       'certainty_state',
                       'true_positive_rate',
                       'true_negative_rate']

        data_frame_numpy = []
        for _ in range(n_iter):
            batch = next(data_gen_val_preprocessed)
            # batch = pre_processor.batch_preprocess(batch)
            for j in range(len(batch[0])):
                data_featurs = []
                print(batch[1].shape)
                # y_true = batch[1][j].reshape((1, self.input_h, self.input_w, self.n_channels))
                y_true = np.expand_dims(batch[1][j], axis=-1)
                # y_pred = model.predict(
                # batch[0][j].reshape((1, self.input_h, self.input_w, self.n_channels)))
                y_pred = model.predict(np.expand_dims(batch[0][j], axis=-1))
                data_featurs.append(float(loss.iou_coef_loss(y_true, y_pred)))
                data_featurs.append(float(loss.dice_coef_loss(y_true, y_pred)))
                data_featurs.append(float(loss.soft_dice_loss(y_true, y_pred)))
                data_featurs.append(float(metric.get_iou_coef()(y_true, y_pred)))
                data_featurs.append(float(metric.get_soft_iou()(y_true, y_pred)))
                data_featurs.append(float(metric.get_dice_coeff()(y_true, y_pred)))
                data_featurs.append(float(metric.get_soft_dice()(y_true, y_pred)))
                data_featurs.append(float(metric.get_mad(self.input_w, self.input_h)(y_true, y_pred)))
                data_featurs.append(float(metric.get_hausdorff_distance(self.input_w, self.input_h)(y_true, y_pred)))
                data_featurs.append(self._model_certainty(y_true, y_pred)[0])
                data_featurs.append(self._model_certainty(y_true, y_pred)[1])
                data_featurs.append(self._model_certainty(y_true, y_pred)[2])
                data_featurs.append(self._model_certainty(y_true, y_pred)[3])
                data_featurs.append(self.true_positive_rate(y_true, y_pred))
                data_featurs.append(self.true_negative_rate(y_true, y_pred))
                data_frame_numpy.append(data_featurs)

        return pd.DataFrame(data_frame_numpy, columns=new_columns, index=val_data_indexes)

    @staticmethod
    def _model_certainty(y_true, y_pred):

        """
        :param y_true: ground truth
        :param y_pred: prediction of the model
        :return: the certainty of the model of every data
        """

        dif = np.abs(y_true - y_pred)
        true_certainty = np.count_nonzero(dif < 0.3) / y_true.size
        false_certainty = np.count_nonzero(dif > 0.7) / y_true.size
        ambiguous = np.count_nonzero(np.logical_and(0.3 <= dif, dif <= 0.7)) / y_true.size
        certainty_list = [true_certainty, false_certainty, ambiguous]
        if np.argmax(certainty_list) == 0:
            return [true_certainty, false_certainty, ambiguous, 'true_certain']
        elif np.argmax(certainty_list) == 1:
            return [true_certainty, false_certainty, ambiguous, 'false_certain']
        elif np.argmax(certainty_list) == 2:
            return [true_certainty, false_certainty, ambiguous, 'ambigous']

    @staticmethod
    def true_positive_rate(y_true, y_pred):

        """
        :param y_true: ground truth
        :param y_pred: prediction of the model
        :return: the percentage of the true positives
        """

        tp = 0
        for i in range(len(y_true[0])):
            for j in range(len(y_true[0][0])):
                if y_pred[0][i, j] == y_true[0][i, j] and y_true[0][i, j] == 1:
                    tp += 1
        return (tp / np.count_nonzero(y_true > 0.5)) * 100

    @staticmethod
    def true_negative_rate(y_true, y_pred):
        """
        :param y_true: ground truth
        :param y_pred: prediction of the model
        :return: the percentage of the true negative
        """
        tn = 0
        for i in range(len(y_true[0])):
            for j in range(len(y_true[0][0])):
                if y_pred[0][i, j] == y_true[0][i, j] and y_true[0][i, j] == 0:
                    tn += 1
        return (tn / np.count_nonzero(y_true > 0.5)) * 100

    # def _model_certainty_v2(self, y_true, y_pred, upper=0.75, lower=0.25):
    #
    #     """Returns only true-certainty: closer to 1 -> model is confident in """
    #
    #     correct_mask = (y_true > 0.5) == (y_pred > 0.5)
    #     correct_predictions = tf.math.count_nonzero(correct_mask)
    #
    #     y_pred_correct = y_pred * tf.cast(correct_mask, tf.float32)
    #     y_pred_correct_certain = y_pred_correct >

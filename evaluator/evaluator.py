import sys
sys.path.append('..')

import pathlib

import numpy as np
import pandas as pd
import tensorflow as tf

from model import metric
from model import loss
from utils.handling_yaml import load_config_file
from .evaluator_base import EvaluatorBase

from tqdm import tqdm


class Evaluator(EvaluatorBase):
    """
    This class is for evaluating the model and data according to the metrics and losses

    Example::

        eval = Evaluator(config)
        df = eval.build_data_frame(model,data_generator,n_iter)

        # Or use exported folder
        eval_report = eval.generate_report(exported_dir)

    """

    def __init__(self):

        """

        """

        super().__init__()

    def build_data_frame(self, model, data_gen_val_preprocessed, n_iter, val_data_indexes):

        """Generates a report as a pandas.DataFrame

        :param val_data_indexes: index of validation data. this will be the index column of the result dataframe
        :param model: the input model which is being evaluated
        :param data_gen_val_preprocessed: the data generator which is being evaluated,
         it has to be a pre-processed data generator
        :param n_iter: number of iterations of the data generator

        :return df: the dataframe which consists of the metrics and losses and also the certainty of
         the model and true positive rate and true negative rate
        """

        # building the dataframe
        new_columns = ['iou_coef_loss',
                       'soft_iou_loss',
                       'dice_coef_loss',
                       'soft_dice_loss',
                       'iou_coef',
                       'soft_iou_coef',
                       'dice_coef',
                       'soft_dice',
                       # 'mad',
                       # 'hausdorff',
                       'truecertainty',
                       'falsecertainty',
                       'ambigous',
                       'certainty_state',
                       'true_positive_rate_0.5',
                       'true_negative_rate_0.5',
                       'true_positive_rate_0.1',
                       'true_negative_rate_0.1',
                       'true_positive_rate_0.3',
                       'true_negative_rate_0.3',
                       'true_positive_rate_0.7',
                       'true_negative_rate_0.7',
                       'true_positive_rate_0.9',
                       'true_negative_rate_0.9']

        data_frame_numpy = []

        with tqdm(total=len(val_data_indexes)) as pbar:
            for _ in range(n_iter):
                x_b, y_b = next(data_gen_val_preprocessed)
                # batch = pre_processor.batch_preprocess(batch)

                processed_batch = model.predict(x_b)
                for y_pred, y_true in zip(processed_batch, y_b):
                    data_featurs = []
                    input_h, input_w, _ = y_pred.shape
                    # y_true = batch[1][j].reshape((1, self.input_h, self.input_w, self.n_channels))
                    # y_true = np.expand_dims(batch[1][j], axis=0)
                    # y_pred = model.predict(
                    # batch[0][j].reshape((1, self.input_h, self.input_w, self.n_channels)))
                    # y_pred = model.predict(np.expand_dims(batch[0][j], axis=0))

                    data_featurs.append(float(loss.iou_coef_loss(y_true, y_pred)))
                    data_featurs.append(float(loss.soft_iou_loss(y_true, y_pred)))
                    data_featurs.append(float(loss.dice_coef_loss(y_true, y_pred)))
                    data_featurs.append(float(loss.soft_dice_loss(y_true, y_pred)))
                    data_featurs.append(float(metric.get_iou_coef()(y_true, y_pred)))
                    data_featurs.append(float(metric.get_soft_iou()(y_true, y_pred)))
                    data_featurs.append(float(metric.get_dice_coeff()(y_true, y_pred)))
                    data_featurs.append(float(metric.get_soft_dice()(y_true, y_pred)))
                    # data_featurs.append(float(metric.get_mad(input_w, input_h)(y_true, y_pred)))
                    # data_featurs.append(float(metric.get_hausdorff_distance(input_w, input_h)(y_true, y_pred)))

                    certainty = self.model_certainty(y_true, y_pred)
                    data_featurs.append(certainty[0])
                    data_featurs.append(certainty[1])
                    data_featurs.append(certainty[2])
                    data_featurs.append(certainty[3])

                    data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.5))
                    data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.5))

                    data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.1))
                    data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.1))

                    data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.3))
                    data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.3))

                    data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.7))
                    data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.7))

                    data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.9))
                    data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.9))

                    data_frame_numpy.append(data_featurs)
                    pbar.update(1)

        # with tqdm(total=len(val_data_indexes)) as pbar:
        #     for _ in range(n_iter):
        #         batch = next(data_gen_val_preprocessed)
        #         # batch = pre_processor.batch_preprocess(batch)
        #
        #         batch_processed = model.predict(batch)
        #         for j in range(len(batch[0])):
        #             data_featurs = []
        #             _, input_h, input_w, _ = batch[1].shape
        #             # y_true = batch[1][j].reshape((1, self.input_h, self.input_w, self.n_channels))
        #             y_true = np.expand_dims(batch[1][j], axis=0)
        #             # y_pred = model.predict(
        #             # batch[0][j].reshape((1, self.input_h, self.input_w, self.n_channels)))
        #             y_pred = model.predict(np.expand_dims(batch[0][j], axis=0))
        #
        #             data_featurs.append(float(loss.iou_coef_loss(y_true, y_pred)))
        #             data_featurs.append(float(loss.soft_iou_loss(y_true, y_pred)))
        #             data_featurs.append(float(loss.dice_coef_loss(y_true, y_pred)))
        #             data_featurs.append(float(loss.soft_dice_loss(y_true, y_pred)))
        #             data_featurs.append(float(metric.get_iou_coef()(y_true, y_pred)))
        #             data_featurs.append(float(metric.get_soft_iou()(y_true, y_pred)))
        #             data_featurs.append(float(metric.get_dice_coeff()(y_true, y_pred)))
        #             data_featurs.append(float(metric.get_soft_dice()(y_true, y_pred)))
        #             # data_featurs.append(float(metric.get_mad(input_w, input_h)(y_true, y_pred)))
        #             # data_featurs.append(float(metric.get_hausdorff_distance(input_w, input_h)(y_true, y_pred)))
        #
        #             certainty = self.model_certainty(y_true, y_pred)
        #             data_featurs.append(certainty[0])
        #             data_featurs.append(certainty[1])
        #             data_featurs.append(certainty[2])
        #             data_featurs.append(certainty[3])
        #
        #             data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.5))
        #             data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.5))
        #
        #             data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.1))
        #             data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.1))
        #
        #             data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.3))
        #             data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.3))
        #
        #             data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.7))
        #             data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.7))
        #
        #             data_featurs.append(self.true_positive_rate(y_true, y_pred, 0.9))
        #             data_featurs.append(self.true_negative_rate(y_true, y_pred, 0.9))
        #
        #             data_frame_numpy.append(data_featurs)
        #             pbar.update(1)

        return pd.DataFrame(data_frame_numpy, columns=new_columns, index=val_data_indexes)

    def generate_report(self, exported_dir):
        config_path = list(pathlib.Path(exported_dir).glob('*.yaml'))[0]
        config = load_config_file(config_path)

        val_data_gen, n_iter_val, val_df = self._create_val_data_gen(config)
        inference_model = self._load_model(config, exported_dir)

        eval_report = self.build_data_frame(inference_model, val_data_gen, n_iter_val, val_df.index)
        return eval_report

    @staticmethod
    def model_certainty(y_true, y_pred):
        """Calculates certainty about the given input

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
        else:
            raise Exception()

    @staticmethod
    def get_conf_mat_elements(y_true, y_pred, threshold=0.5):

        """Returns true positives count

        :param y_true: tensor of shape(1, input_h, input_w, 1).dtype(tf.float32) and {0, 1}
        :param y_pred: tensor of shape(1, input_h, input_w, 1).dtype(tf.float32) and [0, 1]
        :param threshold: threshold on ``y_pred``

        :returns tp: true positives count
        :returns tn: true negatives count
        :returns fp: false positives count
        :returns fn: false negatives count
        """

        y_pred_thresholded = tf.cast(y_pred > threshold, tf.float32)
        y_true_thresholded = tf.cast(y_true > threshold, tf.float32)

        conf_mat = tf.math.confusion_matrix(tf.reshape(y_true_thresholded, -1), tf.reshape(y_pred_thresholded, -1))
        tn, fp, fn, tp = tf.reshape(conf_mat, -1)
        return tp, tn, fp, fn

    def true_positive_rate(self, y_true, y_pred, threshold=0.5):
        """Calculates true positive rate

        :param y_true: ground truth
        :param y_pred: prediction of the model
        :param threshold: threshold on ``y_pred``

        :return: the percentage of the true positives
        """

        tp, tn, fp, fn = self.get_conf_mat_elements(y_true, y_pred, threshold)
        return float((tp / (tp + fn)) * 100)

        # tp = 0
        # for i in range(len(y_true[0])):
        #     for j in range(len(y_true[0][0])):
        #         if y_pred[0][i, j] == y_true[0][i, j] and y_true[0][i, j] == 1:
        #             tp += 1
        # return (tp / np.count_nonzero(y_true > 0.5)) * 100

    def true_negative_rate(self, y_true, y_pred, threshold=0.5):
        """Calculates true negative rate

        :param y_true: ground truth
        :param y_pred: prediction of the model
        :param threshold: threshold on ``y_pred``

        :return: the percentage of the true negative
        """

        tp, tn, fp, fn = self.get_conf_mat_elements(y_true, y_pred, threshold)
        return float((tn / (fp + tn)) * 100)

        # tn = 0
        # for i in range(len(y_true[0])):
        #     for j in range(len(y_true[0][0])):
        #         if y_pred[0][i, j] == y_true[0][i, j] and y_true[0][i, j] == 0:
        #             tn += 1
        # return (tn / np.count_nonzero(y_true > 0.5)) * 100

    # def _model_certainty_v2(self, y_true, y_pred, upper=0.75, lower=0.25):
    #
    #     """Returns only true-certainty: closer to 1 -> model is confident in """
    #
    #     correct_mask = (y_true > 0.5) == (y_pred > 0.5)
    #     correct_predictions = tf.math.count_nonzero(correct_mask)
    #
    #     y_pred_correct = y_pred * tf.cast(correct_mask, tf.float32)
    #     y_pred_correct_certain = y_pred_correct >

import sys
sys.path.append('..')

from .model import metric
from .model import loss
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
        self.threshold=0.5

    def build_data_frame(self, model, data_gen, n_iter):
        """

        :param model: the input model which is being evaluated
        :param data_gen: the data generator which is being evaluated
        :param n_iter: number of iterations of the data generator
        :return: the dataframe which consists of the metrics and losses and also the certainty of
        the model and true positive rate and true negative rate
        """
        # building the dataframe
        data_frame_numpy = []
        for i in range(n_iter):
            each_batch = next(data_gen)
            for j in range(len(each_batch[0])):
                data_featurs = []
                print(each_batch[1].shape)
                y_true = each_batch[1][j].reshape((1, self.input_h, self.input_w, self.n_channels))
                y_pred = model.predict(each_batch[0][j].reshape((1, self.input_h, self.input_w, self.n_channels)))
                data_featurs.append(float(loss.iou_coef_loss(y_true, y_pred)))
                data_featurs.append(float(loss.soft_iou_loss(y_true, y_pred)))
                data_featurs.append(float(loss.dice_coef_loss(y_true, y_pred)))
                data_featurs.append(float(loss.soft_dice_loss(y_true, y_pred)))
                iou_coef = metric.get_iou_coef()
                soft_iou = metric.get_soft_iou()
                dice_coef = metric.get_dice_coeff()
                soft_dice = metric.get_soft_dice()
                hausdorff_distance0 = metric.get_hausdorff_distance(self.input_w, self.input_h)
                hausdorff_distance1 = hausdorff_distance0.hausdorff_distance(y_true, y_pred)
                hausdorff_distance = hausdorff_distance1.hausdorff(y_true, y_pred)
                mad0 = metric.get_mad(self.input_w, self.input_h)
                mad1 = mad0.mad_distance(y_true, y_pred)
                mad = mad1.mad(y_true, y_pred)
                data_featurs.append(float(iou_coef(y_true, y_pred)))
                data_featurs.append(float(soft_iou(y_true, y_pred)))
                data_featurs.append(float(dice_coef(y_true, y_pred)))
                data_featurs.append(float(soft_dice(y_true, y_pred)))
                data_featurs.append(float(hausdorff_distance(y_true, y_pred)))
                data_featurs.append(float(mad(y_true, y_pred)))
                data_featurs.append(self._model_certainty(y_true, y_pred)[0])
                data_featurs.append(self._model_certainty(y_true, y_pred)[1])
                data_featurs.append(self._model_certainty(y_true, y_pred)[2])
                data_featurs.append(self._model_certainty(y_true, y_pred)[3])
                data_featurs.append(self.true_positive_rate(y_true, y_pred))
                data_featurs.append(self.true_negative_rate(y_true, y_pred))
                data_frame_numpy.append(data_featurs)
        return pd.DataFrame(data_frame_numpy, columns=['iou_coef_loss', 'dice_coef_loss',
                                                       'soft_dice_loss', 'iou_coef', 'soft_iou_loss','dice_coef',
                                                       'soft_dice', 'hausdorff_distance','mean_absolute_distance',
                                                       'truecertainty', 'falsecertainty', 'ambigous', 'certainty_state',
                                                       'true_positive_rate', 'true_negative_rate'])

    @staticmethod
    def _model_certainty(y_true, y_pred):
        """
        :param y_true: ground truth
        :param y_pred: prediction of the model
        :return: the certainty of the model of every data
        """
        dif = np.abs(y_true- y_pred)
        truecertainty = np.count_nonzero(dif < 0.3)/y_true.size
        falsecertainty = np.count_nonzero(dif > 0.7)/y_true.size
        ambiguous = np.count_nonzero(np.logical_and(0.3 <= dif, dif <= 0.7))/y_true.size
        certainty_list = [truecertainty, falsecertainty, ambiguous]
        if np.argmax(certainty_list) == 0:
            return [truecertainty, falsecertainty, ambiguous, 'true_certain']
        elif np.argmax(certainty_list) == 1:
            return [truecertainty, falsecertainty, ambiguous, 'false_certain']
        elif np.argmax(certainty_list) == 2:
            return [truecertainty, falsecertainty, ambiguous, 'ambigous']

    def true_positive_rate(self, y_true, y_pred):
        """
        :param y_true: ground truth
        :param y_pred: prediction of the model
        :return: the percentage of the true positives
        """
        TP = 0
        y_pred = y_pred>self.threshold
        for i in range(len(y_true[0])):
            for j in range(len(y_true[0][0])):
                if y_pred[0][i, j] == y_true[0][i, j] and y_true[0][i, j] == 1:
                    TP += 1
        return (TP / np.count_nonzero(y_true > 0.5)) * 100

    def true_negative_rate(self, y_true, y_pred):
        """
        :param y_true: ground truth
        :param y_pred: prediction of the model
        :return: the percentage of the true negative
        """
        TN = 0
        y_pred = y_pred > self.threshold
        for i in range(len(y_true[0])):
            for j in range(len(y_true[0][0])):
                if y_pred[0][i, j] == y_true[0][i, j] and y_true[0][i, j] == 0:
                    TN += 1
        return (TN / np.count_nonzero(y_true < 0.5)) * 100


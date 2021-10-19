import numpy as np
import tensorflow as tf


def get_true_certainty(l_th):

    def true_certainty(y_true, y_pred):
        dif = np.abs(y_true - y_pred)
        tc = np.count_nonzero(dif < l_th) / y_true.size
        return tc

    f = true_certainty
    true_certainty.__name__ += f'_lth{l_th}'

    return f


def get_false_certainty(u_th):

    def false_certainty(y_true, y_pred):
        dif = np.abs(y_true - y_pred)
        fc = np.count_nonzero(dif > u_th) / y_true.size
        return fc

    f = false_certainty
    f.__name__ += f'_uth{u_th}'

    return f


def get_ambiguity(l_th, u_th):

    def ambiguity(y_true, y_pred):
        dif = np.abs(y_true - y_pred)
        log_and = np.logical_and(l_th <= dif, dif <= u_th)
        amb = np.count_nonzero(log_and) / y_true.size
        return amb

    f = ambiguity
    f.__name__ += f'_lth{l_th}_uth{u_th}'

    return ambiguity


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


def get_tpr(threshold=0.5):

    def true_positive_rate(y_true, y_pred):

        """Calculates true positive rate

        :param y_true: ground truth
        :param y_pred: prediction of the model
        :param threshold: threshold on ``y_pred``

        :return: the percentage of the true positives
        """

        tp, tn, fp, fn = get_conf_mat_elements(y_true, y_pred, threshold)
        return float((tp / (tp + fn)) * 100)

    f = true_positive_rate
    f.__name__ += f'_th{threshold}'

    return f


def get_tnr(threshold=0.5):

    def true_negative_rate(y_true, y_pred):

        """Calculates true negative rate

        :param y_true: ground truth
        :param y_pred: prediction of the model
        :param threshold: threshold on ``y_pred``

        :return: the percentage of the true negative
        """

        tp, tn, fp, fn = get_conf_mat_elements(y_true, y_pred, threshold)
        return float((tn / (fp + tn)) * 100)

    f = true_negative_rate
    f.__name__ += f'_th{threshold}'

    return true_negative_rate


# def _model_certainty_v2(self, y_true, y_pred, upper=0.75, lower=0.25):
#
#     """Returns only true-certainty: closer to 1 -> model is confident in """
#
#     correct_mask = (y_true > 0.5) == (y_pred > 0.5)
#     correct_predictions = tf.math.count_nonzero(correct_mask)
#
#     y_pred_correct = y_pred * tf.cast(correct_mask, tf.float32)
#     y_pred_correct_certain = y_pred_correct >

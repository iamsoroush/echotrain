from .metric import *
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff


def example_tensor_data():

    """Returns 2*3*2*1 tensors"""

    y_true = np.array([[[1., 1., 1.], [1., 0., 1.]], [[1., 1., 1.], [1., 0., 1.]]])
    y_pred = np.array([[[0.1, 0.1, 0.1], [0.8, 0.1, 0.8]], [[0.8, 0.8, 0.8], [0.8, 0.1, 0.8]]])
    y_true, y_pred = np.expand_dims(y_true, -1), np.expand_dims(y_pred, -1)

    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    return y_true, y_pred


def manual_hd(y_true, y_pred, threshold=0.5):
    result_list = []
    y_pred = K.cast(y_pred > threshold, 'float32')
    for y, x in y_true, y_pred:
        y, x = np.squeeze(y, 2), np.squeeze(x, 2)
        result = max(directed_hausdorff(y, x)[0], directed_hausdorff(x, y)[0])
        result_list.append(result)

    return sum(result_list) / len(result_list)


def manual_iou_coef(y_true, y_pre, threshold=0.5):
    y_pre = K.cast(y_pre > threshold, 'float32')
    intersection = np.logical_and(y_true, y_pre)
    union = np.logical_or(y_true, y_pre)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def manual_dice_coef(y_true , y_pre, threshlod=0.5):
    y_pred = K.cast(y_pre > threshlod, 'float32')
    intersection = np.logical_and(y_true, y_pred)
    intersection2 = 2 * np.sum(intersection)
    dice_score = intersection2 / np.sum(y_pred + y_true)
    return dice_score


def test_iou_coef():
    iou_metric = get_iou_coef()

    y_true, y_pred = example_tensor_data()

    iou_value = iou_metric(y_true, y_pred)
    iou_true = manual_iou_coef(y_true, y_pred)

    # iou_coef = metric.iou_coef(y_true_1, y_pred_1)
    assert K.abs(iou_value - iou_true) < 0.001

def test_dice_coef():
    y_true = np.array([[1., 1., 1.],
                     [1., 0., 1.],
                     [1., 1., 0.]])

    y_pred = np.array([[0.1, 0.1, 0.1],
                     [0.8, 0.1, 0.8],
                     [0.8, 0.8, 0.1]])
    y_true_1, y_pred_1 = np.expand_dims(y_true, 0), np.expand_dims(y_pred, 0)
    y_true_1, y_pred_1 = np.expand_dims(y_true_1, -1), np.expand_dims(y_pred_1, -1)
    iou_coef1 = iou_coef(y_true_1,y_pred_1)
    assert K.abs(float(iou_coef1) - manual_iou_coef(y_true , y_pred)) < 0.0001

def test_hausdorff_distance():
    y_true, y_pred = example_tensor_data()

    hd_result = get_hausdorff_distance(y_true.shape[1], y_true.shape[2])(y_true, y_pred)
    assert K.abs(hd_result - manual_hd(y_true, y_pred)) < 0.001

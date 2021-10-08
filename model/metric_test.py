import metric
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff


def manual_hd(y_true,y_pred):
    threshlod = 0.5
    y_pred = K.cast(y_pred > threshlod, 'float32')
    result = max(directed_hausdorff(y_true, y_pred)[0], directed_hausdorff(y_pred, y_true)[0])
    return result

def manual_iou_coef(y_true , y_pre ):
    threshlod = 0.5
    y_pre = K.cast(y_pre > threshlod, 'float32')
    intersection = np.logical_and(y_true, y_pre)
    union = np.logical_or(y_true, y_pre)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def manual_dice_coef(y_true , y_pre ):
    threshlod = 0.5
    y_pre = K.cast(y_pre > threshlod, 'float32')
    intersection = np.logical_and(y_true, y_pre)
    intersection2 = 2 * np.sum(intersection)
    dice_score = intersection2/np.sum(y_pre+ y_true)
    return dice_score



def test_iou_coef():
    y_true = np.array([[1., 1., 1.],
                     [1., 0., 1.],
                     [1., 1., 0.]])

    y_pred = np.array([[0.1, 0.1, 0.1],
                     [0.8, 0.1, 0.8],
                     [0.8, 0.8, 0.1]])

    y_true_1, y_pred_1 = np.expand_dims(y_true, 0), np.expand_dims(y_pred, 0)
    y_true_1, y_pred_1 = np.expand_dims(y_true_1, -1), np.expand_dims(y_pred_1, -1)
    iou_coef = metric.iou_coef(y_true_1,y_pred_1)
    assert K.abs(float(iou_coef) - manual_iou_coef(y_true , y_pred)) < 0.00001

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
    y_true = np.array([[1., 1., 1.],
                       [1., 0., 1.],
                       [1., 1., 0.]])

    y_pred = np.array([[0.1, 0.1, 0.1],
                       [0.8, 0.1, 0.8],
                       [0.8, 0.8, 0.1]])

    y_true_1, y_pred_1 = np.expand_dims(y_true, 0), np.expand_dims(y_pred, 0)
    y_true_1, y_pred_1 = np.expand_dims(y_true_1, -1), np.expand_dims(y_pred_1, -1)
    y_true_1 = tf.convert_to_tensor(y_true_1, dtype=tf.float32)
    y_pred_1 = tf.convert_to_tensor(y_pred_1, dtype=tf.float32)
    hd_result = metric.hausdorff_distance(y_true_1 , y_pred_1 )
    assert K.abs(float(hd_result) - manual_hd(y_true, y_pred)) < 0.00001


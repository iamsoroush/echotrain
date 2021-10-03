import metric
import numpy as np
from tensorflow.keras import backend as K


def manual_iou_coef(y_true , y_pre ):
    threshlod = 0.5
    y_pre = K.cast(y_pre > threshlod, 'float32')
    intersection = np.logical_and(y_true, y_pre)
    union = np.logical_or(y_true, y_pre)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score



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
    assert float(iou_coef) == manual_iou_coef(y_true , y_pred)


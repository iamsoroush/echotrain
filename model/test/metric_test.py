from echotrain.model.metric import *
import numpy as np
import pytest
from tensorflow.keras import backend as K
import tensorflow as tf
from scipy.spatial.distance import directed_hausdorff


class TestClass:

    @pytest.fixture
    def tensor_data_test_case(self):
        """Returns 1*3*3*1 tensors"""

        y_true = np.array([[1., 1., 1.], [1., 0., 1.], [0., 0., 0.]])
        y_pred = np.array([[0.6, 0.9, 0.7], [0.8, 0.1, 0.8], [0.1, 0.2, 0.3]])
        y_true, y_pred = np.expand_dims(y_true, 0), np.expand_dims(y_pred, 0)
        y_true, y_pred = np.expand_dims(y_true, -1), np.expand_dims(y_pred, -1)

        y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        return y_true, y_pred

    @staticmethod
    def manual_hd(y_true, y_pred, threshold=0.5):
        result_list = []
        y_pred = K.cast(y_pred > threshold, 'float32')
        for y, x in zip(y_true, y_pred):
            y, x = np.squeeze(y, 2), np.squeeze(x, 2)
            result = max(directed_hausdorff(y, x)[0], directed_hausdorff(x, y)[0])
            result_list.append(result)

        return sum(result_list) / len(result_list)

    @staticmethod
    def manual_iou_coef(y_true, y_pre, threshold=0.5):
        y_pre = K.cast(y_pre > threshold, 'float32')
        intersection = np.logical_and(y_true, y_pre)
        union = np.logical_or(y_true, y_pre)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    @staticmethod
    def manual_dice_coef(y_true, y_pre, threshlod=0.5):
        y_pred = K.cast(y_pre > threshlod, 'float32')
        intersection = np.logical_and(y_true, y_pred)
        intersection2 = 2 * np.sum(intersection)
        dice_score = intersection2 / np.sum(y_pred + y_true)
        return dice_score

    @pytest.mark.parametrize("threshold, smooth", [
        (0.5, 0.001)
    ])
    def test_iou_coef(self, tensor_data_test_case, threshold, smooth):
        y_true, y_pred = tensor_data_test_case

        iou_value = get_iou_coef(threshold, smooth)(y_true, y_pred)
        iou_true = self.manual_iou_coef(y_true, y_pred)
        print('iou_coef:', iou_value)
        assert iou_value.shape == ([1])
        assert 'float' in str(iou_value.dtype)
        assert K.abs(iou_value - iou_true) < 0.001

    @pytest.mark.parametrize("threshold, smooth", [
        (0.5, 0.001)
    ])
    def test_dice_coef(self, tensor_data_test_case, threshold, smooth):
        y_true, y_pred = tensor_data_test_case

        dice_value = get_dice_coeff(threshold, smooth)(y_true, y_pred)
        dice_true = self.manual_dice_coef(y_true, y_pred)
        print('dice_coef:', dice_value)
        assert dice_value.shape == ([1])
        assert 'float' in str(dice_value.dtype)
        assert K.abs(dice_true - dice_value) < 0.1

    @pytest.mark.parametrize("w, h", [
        (3, 3)
    ])
    def test_hausdorff_distance(self, tensor_data_test_case, w, h):
        y_true, y_pred = tensor_data_test_case
        hd_result = get_hausdorff_distance(w, h)(y_true, y_pred)
        print('hausdorff_distance:', hd_result)

        assert hd_result.shape == ()
        assert 'float' in str(hd_result.dtype)
        assert K.abs(hd_result - self.manual_hd(y_true, y_pred)) < 0.001

    @pytest.mark.parametrize("w, h", [
        (3, 3)
    ])
    def test_get_mad(self, tensor_data_test_case, w, h):
        y_true, y_pred = tensor_data_test_case
        mad = get_mad(w, h)(y_true, y_pred)
        print('mad:', mad)
        assert mad.shape == ()
        assert 'float' in str(mad.dtype)

    @pytest.mark.parametrize("epsilon", [
        1e-6
    ])
    def test_get_soft_dice(self, tensor_data_test_case, epsilon):
        y_true, y_pred = tensor_data_test_case

        soft_dice = get_soft_dice(epsilon)(y_true, y_pred)
        print('soft_dice:', soft_dice)
        assert soft_dice.shape == ()
        assert 'float' in str(soft_dice.dtype)

    @pytest.mark.parametrize("smooth", [
        0.001
    ])
    def test_get_soft_iou(self, tensor_data_test_case, smooth):
        y_true, y_pred = tensor_data_test_case

        soft_iou = get_soft_iou(smooth)(y_true, y_pred)
        print('soft_dice:', soft_iou)
        assert soft_iou.shape == ([1])
        assert 'float' in str(soft_iou.dtype)

from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
from sklearn.utils.extmath import cartesian
import math


def iou_coef(y_true, y_pred, threshlod=0.5):
    """
    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :param smooth:
    :return:calculate Intersection over Union for y_true and y_pred
    the input shape should be (b , w , h , n)
    """

    #  keras uses float32 instead of float64,
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # our input y_pred is softmax prediction so we change it to 0 ,1 classes
    y_pred_thresholded = K.cast(y_pred > threshlod, tf.float32)

    # axis depends on input shape
    axis = [1, 2, 3]

    smooth = .001

    intersection = K.sum(K.abs(y_true * y_pred_thresholded), axis=axis)
    union = K.sum(y_true, axis=axis) + K.sum(y_pred_thresholded, axis=axis) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def soft_iou(y_true, y_pred):
    """
    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :param smooth:
    :return:calculate Intersection over Union for y_true and y_pred
    the input shape should be (b , w , h , n)
    """

    #  keras uses float32 instead of float64,
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    # axis depends on input shape
    axis = [1, 2, 3]

    smooth = .001

    intersection = K.sum(K.abs(y_true * y_pred), axis=axis)
    union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred):
    """

    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :param smooth:
    :return: calculate dice coefficient between y_true and y_pred
    """
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')

    smooth = .001
    threshold = 0.5

    y_pred_thresholded = K.cast(y_pred > threshold, tf.float32)

    intersection = K.sum(y_true * y_pred_thresholded, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred_thresholded, axis=[1, 2])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def soft_dice(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    epsilon = 1e-6
    """
        Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the channels_last format.

        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
            epsilon: Used for numerical stability to avoid divide by zero errors
        """
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)

    soft_dice = K.mean((numerator + epsilon) / (denominator + epsilon))
    return soft_dice


def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D

def hausdorff(w, h):

    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w),
                                                        np.arange(h)]), dtype=tf.float32)

    def hausdorff_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        threshlod=0.5

        y_pred = K.cast(y_pred > threshlod, tf.float32)
        def loss(y_true, y_pred):
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            y_pred = K.flatten(y_pred)
            p = y_pred
            d_matrix = cdist(all_img_locations, gt_points)
            k_min = tf.cast(K.min(d_matrix, 1), 'float32')
            p_k_min = p * k_min
            k_max = K.max(p_k_min)
            return float(k_max)

        batched_losses_1 = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)

        batched_losses_2 = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_pred, y_true),
                                   dtype=tf.float32)

        stacked  = tf.stack([batched_losses_1, batched_losses_2])

        return K.mean(K.max(stacked, axis = 0))
    return hausdorff_loss

def Mad(w, h):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w),
                                                        np.arange(h)]), dtype=tf.float32)

    def mad_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        threshlod=0.5

        y_pred = K.cast(y_pred > threshlod, tf.float32)
        def m_loss(y_true, y_pred):
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            y_pred = K.flatten(y_pred)
            p = y_pred
            d_matrix = cdist(all_img_locations, gt_points)
            k_min = tf.cast(K.min(d_matrix, 1), 'float32')
            p_k_min = p * k_min
            k_mean = K.mean(p_k_min)
            return float(k_mean)

        batched_losses_1 = tf.map_fn(lambda x:
                                   m_loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)

        batched_losses_2 = tf.map_fn(lambda x:
                                   m_loss(x[0], x[1]),
                                   (y_pred, y_true),
                                   dtype=tf.float32)

        stacked  = tf.stack([batched_losses_1, batched_losses_2])

        return K.mean(stacked)
    return mad_loss

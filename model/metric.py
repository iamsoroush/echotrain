from tensorflow.keras import backend as K
import tensorflow as tf


def get_iou_coef(threshold=0.5, smooth=0.001):

    def iou_coef(y_true, y_pred):
        """
        :param y_true: label image from the dataset
        :param y_pred: model segmented image prediction
        :return:calculate Intersection over Union for y_true and y_pred
        the input shape should be (b , w , h , n)
        """

        #  keras uses float32 instead of float64,
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')

        # our input y_pred is softmax prediction so we change it to 0 ,1 classes
        y_pred_thresholded = K.cast(y_pred > threshold, tf.float32)

        # axis depends on input shape
        axis = tuple(range(1, len(y_pred.shape) - 1))

        intersection = K.sum(K.abs(y_true * y_pred_thresholded), axis=axis)
        union = K.sum(y_true, axis=axis) + K.sum(y_pred_thresholded, axis=axis) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

    return iou_coef


def get_soft_iou(smooth=0.001):

    def soft_iou(y_true, y_pred):

        """
        :param y_true: label image from the dataset
        :param y_pred: model segmented image prediction
        :return:calculate Intersection over Union for y_true and y_pred
        the input shape should be (b , w , h , n)
        """

        #  keras uses float32 instead of float64,
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')

        axis = tuple(range(1, len(y_pred.shape) - 1))

        intersection = K.sum(K.abs(y_true * y_pred, axis=axis))
        union = K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

    return soft_iou


def get_dice_coeff(threshold=0.5, smooth=0.001):

    def dice_coef(y_true, y_pred):

        """

        :param y_true: label image from the dataset
        :param y_pred: model segmented image prediction
        :return: calculate dice coefficient between y_true and y_pred
        """

        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')

        axis = tuple(range(1, len(y_pred.shape) - 1))

        y_pred_thresholded = K.cast(y_pred > threshold, tf.float32)

        intersection = K.sum(y_true * y_pred_thresholded, axis=axis)
        union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred_thresholded, axis=axis)
        dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
        return dice

    return dice_coef


def get_soft_dice(epsilon=1e-6):

    def soft_dice(y_true, y_pred):

        """Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
        Assumes the channels_last format.

        # Arguments
            y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
            y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax)
            epsilon: Used for numerical stability to avoid divide by zero errors
        """

        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')

        # skip the batch and class axis for calculating Dice score
        axis = tuple(range(1, len(y_pred.shape) - 1))

        numerator = 2. * K.sum(y_pred * y_true, axis)
        denominator = K.sum(K.square(y_pred) + K.square(y_true), axis)

        return K.mean((numerator + epsilon) / (denominator + epsilon))

    return soft_dice

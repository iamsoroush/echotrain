from tensorflow.keras import backend as K
import tensorflow as tf


def iou_coef(y_true, y_pred, smooth=1):
    """
    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :param smooth:
    :return:calculate Intersection over Union for y_true and y_pred
    """

    threshlod = 0.5

    y_pred_thresholded = K.cast(y_pred > threshlod, tf.float32)

    intersection = K.sum(K.abs(y_true * y_pred_thresholded), axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred_thresholded, axis=[1, 2]) - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou

def iou_coef_loss(y_true, y_pred):
    """

   :param y_true: label image from the dataset
   :param y_pred: model segmented image prediction
   :return: iou coefficient loss function
   """

    return 1 - iou_coef(y_true, y_pred)



def dice_coef(y_true, y_pred):
    """

    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :param smooth:
    :return: calculate dice coefficient between y_true and y_pred
    """

    smooth = 1
    threshold = 0.5

    y_pred_thresholded = K.cast(y_pred > threshold, tf.float32)

    intersection = K.sum(y_true * y_pred_thresholded, axis=[1, 2])
    union = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred_thresholded, axis=[1, 2])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice


def dice_coef_loss(y_true, y_pred):
    """

    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :return: dice coefficient loss function
    """
    return 1 - dice_coef(y_true, y_pred)

def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
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

        return 1 - K.mean((numerator + epsilon) / (denominator + epsilon))  # average over classes and batch


def soft_dice(y_pred, y_true):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    intersect = T.sum(y_pred * y_true, 0)
    denominator = T.sum(y_pred, 0) + T.sum(y_true, 0)
    dice_scores = T.constant(2) * intersect / (denominator + T.constant(1e-6))
    return dice_scores

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

def weighted_hausdorff_distance(w, h, alpha):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w),
                                               np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1),
                                                num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss
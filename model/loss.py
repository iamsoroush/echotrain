from .metric import get_iou_coef, get_dice_coeff, get_soft_dice, get_soft_iou


def iou_coef_loss(y_true, y_pred):
    """

   :param y_true: label image from the dataset
   :param y_pred: model segmented image prediction
   :return: iou coefficient loss function
   """

    return 1 - get_iou_coef()(y_true, y_pred)


def soft_iou_loss(y_true, y_pred ):
    return 1 - get_soft_iou()(y_true, y_pred)  # average over classes and batch


def dice_coef_loss(y_true, y_pred):
    """

    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :return: dice coefficient loss function
    """
    return 1 - get_dice_coeff()(y_true, y_pred)


def soft_dice_loss(y_true, y_pred):
    return 1 - get_soft_dice()(y_true, y_pred)  # average over classes and batch

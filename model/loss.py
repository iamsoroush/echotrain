
import metric



def iou_coef_loss(y_true, y_pred):
    """

   :param y_true: label image from the dataset
   :param y_pred: model segmented image prediction
   :return: iou coefficient loss function
   """

    return 1 - metric.iou_coef(y_true, y_pred)

def soft_iou_loss(y_true, y_pred ):


    return 1 - metric.soft_iou(y_true, y_pred )  # average over classes and batch

def dice_coef_loss(y_true, y_pred):
    """

    :param y_true: label image from the dataset
    :param y_pred: model segmented image prediction
    :return: dice coefficient loss function
    """
    return 1 - metric.dice_coef(y_true, y_pred)

def soft_dice_loss(y_true, y_pred ):


    return 1 - metric.soft_dice(y_true, y_pred )  # average over classes and batch
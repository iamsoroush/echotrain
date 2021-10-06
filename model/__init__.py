from .model_base import ModelBase
from .inference_engine import EchoInference
from .pre_processing import PreProcessor
from .unet import UNet


# import metric
# import loss
# import numpy as np
#
# y_true_1 = np.array([[1., 1. , 1.] ,
#                   [1. , 0. , 1.] ,
#                   [1., 1. , 0.]])
#
# y_pred_1 =  np.array([[ 0.1 , 0.8  , 0.8 ] ,
#                       [ 0.8 , 0.1 , 0.8 ] ,
#                       [0.8 ,0.8 , 0.1]])
#
# y_true_1 , y_pred_1= np.expand_dims(y_true_1 , 0), np.expand_dims(y_pred_1 , 0)
# y_true_1 , y_pred_1= np.expand_dims(y_true_1 ,-1), np.expand_dims(y_pred_1 ,-1)
#
# #metric
# dice=metric.dice_coef(y_true_1 , y_pred_1)
# print("dice:" ,float(dice))
#
# soft_dice=metric.soft_dice(y_true_1 , y_pred_1)
# print("soft_dice:" ,float(soft_dice))
#
# iou=metric.iou_coef(y_true_1 , y_pred_1)
# print("iou:" ,float(iou))
#
# soft_iou=metric.soft_iou(y_true_1 , y_pred_1)
# print("soft_iou:" ,float(soft_iou))
#
# #loss
# iou_loss=loss.iou_coef_loss(y_true_1 , y_pred_1)
# print("iou loss:" ,float(iou_loss))
#
# iou_soft_loss=loss.soft_iou_loss(y_true_1 , y_pred_1)
# print("soft iou loss:" ,float(iou_soft_loss))
#
# dice_loss=loss.dice_coef_loss(y_true_1 , y_pred_1)
# print("dice loss:" ,float(dice_loss))
#
# dice_loss_loft=loss.soft_dice_loss(y_true_1 , y_pred_1)
# print("soft dice loss:" ,float(dice_loss_loft))
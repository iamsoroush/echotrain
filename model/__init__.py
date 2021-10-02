from .model_base import ModelBase
from .inference_engine import EchoInference
from .pre_processing import PreProcessor
from .unet import UNet
import metric
import numpy as np

y_true_1 = np.array([[1., 1. , 1.] ,
                  [1. , 0. , 1.] ,
                  [1., 1. , 0.]])

y_pred_1 =  np.array([[ 0.1 , 0.8  , 0.8 ] ,
                      [ 0.8 , 0.1 , 0.8 ] ,
                      [0.8 ,0.8 , 0.1]])

iou_coef=metric.iou_coef(y_true_1, y_pred_1)
print(iou_coef)
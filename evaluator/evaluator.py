from model import metric
from model import loss
import numpy as np
import pandas as pd
import os
import sys
from model.unet import UNet
import tensorflow as tf

utils_dir = os.path.abspath('../utils')
sys.path.append(utils_dir)
from handling_yaml import load_config_file

config_path = "../config/config_example.yaml"
config = load_config_file(config_path)


class Evaluator:
    def __init__(self, config):
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels = config.n_channels

    def build_data_frame(self, model, data_gen, n_iter):
        data_frame_numpy = []
        for i in range(n_iter):
            each_batch = next(data_gen)
            for j in range(len(each_batch[0])):
                data_featurs = []
                y_true = each_batch[1][j].reshape((1,self.input_h,self.input_w,self.n_channels))
                y_pred = model.predict(each_batch[0][j].reshape((1,self.input_h,self.input_w,self.n_channels)))
                data_featurs.append(i)
                data_featurs.append(j)
                data_featurs.append(float(loss.iou_coef_loss(y_true, y_pred)))
                data_featurs.append(float(loss.dice_coef_loss(y_true, y_pred)))
                data_featurs.append(float(loss.soft_dice_loss(y_true, y_pred)))
                data_featurs.append(float(metric.iou_coef(y_true, y_pred)))
                #data_featurs.append(float(metric.soft_iou(y_true, y_pred)))
                data_featurs.append(float(metric.dice_coef(y_true, y_pred)))
                data_featurs.append(float(metric.soft_dice(y_true, y_pred)))
                data_frame_numpy.append(data_featurs)
        return pd.DataFrame(data_frame_numpy, columns=['batch_index','data_index','iou_coef_loss', 'dice_coef_loss'
                                                        ,'soft_dice_loss', 'iou_coef', 'dice_coef','soft_dice'])


def data_gen():
    for i in range(3):
        x = np.random.randn(25, 256, 256, 1)
        y = np.random.randn(25, 256, 256, 1)
        yield x,y


unet = UNet(config)
model = unet.generate_training_model()
e = Evaluator(config)
print(e.build_data_frame(model, data_gen(), 3))



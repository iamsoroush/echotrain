from unet import UNet
import pandas as pd
import numpy as np
import os
import sys
utils_dir = os.path.abspath('../utils')
sys.path.append(utils_dir)
from handling_yaml import load_config_file

config_path = "../config/config_example.yaml"
config = load_config_file(config_path)

class Evaluator:
    def __init__(self, config):
        self.batch_size=config.data_handler.batch_size
        self.input_h=config.input_h
        self.input_w = config.input_w
        self.n_channels=config.n_channels
        self.metrics=config.model.metrics

    def build_data_frame(self, model, data_gen):
        data_frame_numpy=[]
        for i in range(3):#self.batch_size
            each_batch=next(data_gen)
            y_true=each_batch[1][i].reshape(1,self.input_h,self.input_w,self.n_channels)
            y_pred=model.predict(each_batch[0][i].reshape(1,self.input_h,self.input_w,self.n_channels))
            data_frame_numpy.append(model.evaluate(y_true,y_pred))
        return pd.DataFrame(data_frame_numpy,columns=['loss'].append(self.metrics))

def data_gen():
    for i in range(3):
        batch=np.random.randn(2,25,256,256,1)
        yield batch

unet=UNet(config)
model=unet.generate_training_model()
e=Evaluator(config)
print(e.build_data_frame(model,data_gen()))



from model.ejection_fraction.ejection_fraction_base import EFBase
from tensorflow.keras.models import load_model
from dataset.dataset_ef import EFDataset
import os
import shutil
from skimage.measure import regionprops
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import pandas as pd
import skimage.io as io
import pickle
import numpy as np


class EFModel_AL(EFBase):

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self._get_config()

    def area_length_volume(frame):
        """

        Args:
            frame: the segmentation map

        Returns:
            the volume
        """

        label_frame = frame.astype(np.int64).reshape(112, 112)
        rp = [regionprops(label_frame)[0].area,
              regionprops(label_frame)[0].major_axis_length,
              ]
        area = rp[0]
        length = rp[1]
        vol = 0.85 * (area * area) / length
        return vol

    def ef_estimation(self, ed_frame, es_frame):
        ed_vol = self.area_length_volume(ed_frame)
        es_vol = self.area_length_volume(es_frame)
        return float((ed_vol - es_vol) / ed_vol * 100)

    def train(self):
        pass

    def export(self, model):
        pass

    def _load_al_model(self):
        pass

    def _load_al_to_vol_model(self):
        pass

    def _get_config(self):
        """

        Get needed information from config.yaml file

        """
        try:
            self.estimation_method = self.config.estimation_method
            if self.estimation_method == "al":
                self.al_address = self.config.al.al_address
                # self.al_to_vol_model_dir = self.config.al.al_to_vol_model_dir
                # self.al_to_vol_model_type = self.config.al.train.model_type
                # self.exported_dir = self.config.exported_dir
            else:
                raise ValueError
        except ValueError:
            print("estimation_method should be 'al'.")

    def _al_to_vol_model(self):
        pass

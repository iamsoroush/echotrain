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

    def ef_estimation(self, ed_frame, es_frame):

        ed_al = self._load_al_model().predict(ed_frame)
        es_al = self._load_al_model().predict(es_frame)
        ed_vol = self._load_al_to_vol_model().predict(ed_al)
        es_vol = self._load_al_to_vol_model().predict(es_al)
        return float((ed_vol - es_vol) / ed_vol * 100)

    def train(self):

        ef_dataset = EFDataset(self.config)
        images, volumes = ef_dataset.volume_dataset('image', 'train')
        al_model = self._load_al_model()
        encoded_images = al_model.predict(images)
        al_to_vol_model = self._al_to_vol_model()
        al_to_vol_model.fit(encoded_images, volumes)
        return al_to_vol_model

    def export(self, model):

        pickle.dump(model, open('en_to_v.sav', 'wb'))
        os.makedirs(os.path.join(self.exported_dir, 'exported'))
        shutil.move('en_to_v.sav', os.path.join(self.exported_dir, 'exported'))

    def _load_al_model(self):
        """

        Returns:load the area-length model that is saved in the encoder_address directory at config.yaml file

        """

        return load_model(self.al_address)

    def _load_al_to_vol_model(self):
        """

        Returns:load model that is saved in al_to_vol_model_dir directory at config.yaml file

        """

        return pickle.load(open(self.al_to_vol_model_dir, 'rb'))

    def _get_config(self):
        """

        Get needed information from config.yaml file

        """
        try:
            self.estimation_method = self.config.estimation_method
            if self.estimation_method == "al":
                self.al_address = self.config.al.al_address
                self.al_to_vol_model_dir = self.config.al.al_to_vol_model_dir
                self.al_to_vol_model_type = self.config.al.train.model_type
                self.exported_dir = self.config.exported_dir
            else:
                raise ValueError
        except ValueError:
            print("estimation_method should be 'al'.")

    def _al_to_vol_model(self):
        """

        Returns:the sklearn model that is designated in model_type section of config.yaml file

        """

        if self.al_to_vol_model_type == 'svr':
            return SVR(kernel='rbf')
        elif self.al_to_vol_model_type == 'rfr':
            return RandomForestRegressor()
        elif self.al_to_vol_model_type == 'knn':
            return KNeighborsRegressor()
        elif self.al_to_vol_model_type == 'gbr':
            return GradientBoostingRegressor()
        elif self.al_to_vol_model_type == 'dtr':
            return DecisionTreeRegressor()
        elif self.al_to_vol_model_type == 'lr':
            return LinearRegression()
        elif self.al_to_vol_model_type == 'mlp':
            return MLPRegressor()
        elif self.al_to_vol_model_type == 'dr':
            return DummyRegressor()
        elif self.al_to_vol_model_type == 'gpr':
            return GaussianProcessRegressor()
        elif self.al_to_vol_model_type == 'sgdr':
            return SGDRegressor()
        elif self.al_to_vol_model_type == 'abr':
            return AdaBoostRegressor()

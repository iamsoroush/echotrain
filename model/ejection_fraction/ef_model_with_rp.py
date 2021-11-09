from .model.ejection_fraction.ejection_fraction_base import EFBase
from skimage.measure import regionprops
from .dataset.dataset_ef import EFDataset
import os
import shutil

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

import numpy as np
import pickle


class EFModel_RP(EFBase):

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self._get_config()

    def ef_estimation(self, ed_frame, es_frame):

        ed_rp = self._frame_to_rp(ed_frame)
        es_rp = self._frame_to_rp(es_frame)
        ed_vol = self._load_rp_to_vol_model().predict(ed_rp)
        es_vol = self._load_rp_to_vol_model().predict(es_rp)
        return (ed_vol - es_vol) / ed_vol * 100

    def train(self):

        ef_dataset = EFDataset(self.config)
        labels, volumes = ef_dataset.volume_dataset('label','train')
        rps = []
        for label in labels:
            rps.append(self._frame_to_rp(label))
        rps = np.array(rps).reshape(-1, 6)
        rp_to_vol_model = self._rp_to_vol_model
        rp_to_vol_model.fit(rps, volumes)
        return rp_to_vol_model

    def export(self,model):

        pickle.dump(model, open('rp_to_v.sav', 'wb'))
        os.makedirs(os.path.join(self.exported_dir, 'exported'))
        shutil.move('rp_to_v.sav', os.path.join(self.exported_dir, 'exported'))


    @staticmethod
    def _frame_to_rp(frame):
        """

        Args:
            frame: numpy frame of the label

        Returns:
            region properties features of the frame in numpy format
            region properties include:
                area
                convex_area
                eccentricity
                major_axis_length
                minor_axis_length
                orientation
        """
        rp = [regionprops(frame.astype(np.int64))[0].area, regionprops(frame.astype(np.int64))[0].convex_area,
              regionprops(frame.astype(np.int64))[0].eccentricity,
              regionprops(frame.astype(np.int64))[0].major_axis_length,
              regionprops(frame.astype(np.int64))[0].minor_axis_length,
              regionprops(frame.astype(np.int64))[0].orientation]
        rp = np.array(rp).reshape(1, -1)
        return rp

    def _load_rp_to_vol_model(self):
        """

        Returns:load model that is saved in rp_to_vol_model_dir directory at config.yaml file

        """

        return pickle.load(open(self.rp_to_vol_model_dir, 'rb'))

    def _get_config(self):
        """

        Get needed information from config.yaml file

        """
        try:
            self.estimation_method = self.config.estimation_method
            if self.estimation_method == "rp":
                self.rp_to_vol_model_dir = self.config.rp.rp_to_vol_model_dir
                self.rp_to_vol_model_type = self.config.rp.train.model_type
                self.exported_dir = self.config.exported_dir
            else:
                raise ValueError
        except ValueError:
            print("estimation_method should be 'rp'.")

    def _rp_to_vol_model(self):

        if self.rp_to_vol_model_type == 'svr':
            return SVR(kernel='rbf')
        elif self.rp_to_vol_model_type == 'rfr':
            return RandomForestRegressor()
        elif self.rp_to_vol_model_type == 'knn':
            return KNeighborsRegressor()
        elif self.rp_to_vol_model_type == 'gbr':
            return GradientBoostingRegressor()
        elif self.rp_to_vol_model_type == 'dtr':
            return DecisionTreeRegressor()
        elif self.rp_to_vol_model_type == 'lr':
            return LinearRegression()
        elif self.rp_to_vol_model_type == 'mlp':
            return MLPRegressor()
        elif self.rp_to_vol_model_type == 'dr':
            return DummyRegressor()
        elif self.rp_to_vol_model_type == 'gpr':
            return GaussianProcessRegressor()
        elif self.rp_to_vol_model_type == 'sgdr':
            return SGDRegressor()
        elif self.rp_to_vol_model_type == 'abr':
            return AdaBoostRegressor()
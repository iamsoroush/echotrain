from model.ejection_fraction.ejection_fraction_base import EFBase
from tensorflow.keras.models import load_model
from .dataset.dataset_ef import EfDataset

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.dummy import DummyRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

import pickle


class EFModel_Encoder(EFBase):

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self._get_config()

    def ef_estimation(self, ed_frame, es_frame):

        ed_en = self._load_encoder_model().predict(ed_frame)
        es_en = self._load_encoder_model().predict(es_frame)
        ed_vol = self._load_en_to_vol_model().predict(ed_en)
        es_vol = self._load_en_to_vol_model().predict(es_en)
        return (ed_vol - es_vol) / ed_vol * 100

    def train(self):

        ef_dataset = EfDataset(self.config)
        labels, volumes = ef_dataset.ef_dataset('image', 'train')
        encoding_model = self._load_encoder_model()
        encoded_images = encoding_model.predict(images)
        en_to_vol_model = self._en_to_vol_model()
        en_to_vol_model.fit(encoded_images, volumes)
        return en_to_vol_model

    def export(self):
        pass

    def _load_encoder_model(self):

        return load_model(self.encoder_address)

    def _load_en_to_vol_model(self):

        return pickle.load(open(self.en_to_vol_model_dir, 'rb'))

    def _get_config(self):

        try:
            self.estimation_method = self.config.estimation_method
            if self.estimation_method == "encoder":
                self.encoder_address = self.config.encoder.encoder_address
                self.en_to_vol_model_dir = self.config.encoder.en_to_vol_model_dir
                self.en_to_vol_model_type = self.config.encoder.train.model_type
                self.exported_dir = self.config.exported_dir
            else:
                raise ValueError
        except ValueError:
            print("estimation_method should be 'encoder'.")

    def _en_to_vol_model(self):

        if self.en_to_vol_model_type == 'svr':
            return SVR(kernel='rbf')
        elif self.en_to_vol_model_type == 'rfr':
            return RandomForestRegressor()
        elif self.en_to_vol_model_type == 'knn':
            return KNeighborsRegressor()
        elif self.en_to_vol_model_type == 'gbr':
            return GradientBoostingRegressor()
        elif self.en_to_vol_model_type == 'dtr':
            return DecisionTreeRegressor()
        elif self.en_to_vol_model_type == 'lr':
            return LinearRegression()
        elif self.en_to_vol_model_type == 'mlp':
            return MLPRegressor()
        elif self.en_to_vol_model_type == 'dr':
            return DummyRegressor()
        elif self.en_to_vol_model_type == 'gpr':
            return GaussianProcessRegressor()
        elif self.en_to_vol_model_type == 'sgdr':
            return SGDRegressor()
        elif self.en_to_vol_model_type == 'abr':
            return AdaBoostRegressor()
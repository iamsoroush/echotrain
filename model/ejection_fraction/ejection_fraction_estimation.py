import cv2
import numpy as np
import pickle

class EFEstimation:

    def __init__(self, config):

        self.config = config
        self.dataset_class = config.dataset_class
        self.batch_size = config.data_handler.batch_size
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels = config.n_channels

    @staticmethod
    def load_model(address):
        return pickle.load(open(address,'rb'))

    def ef_estimation(self, ed_frame, es_frame, model):
        ed_volume = self._area_to_volume_conversion(self._area(ed_frame), model)
        es_volume = self._area_to_volume_conversion(self._area(es_frame), model)

        return (ed_volume - es_volume) / ed_volume

    @staticmethod
    def _area_to_volume_conversion(area, model):
        return model.predict(np.array([area]).reshape(-1,1))

    @staticmethod
    def _area(image):
        return float(cv2.countNonZero(image))




import numpy as np
import pickle
from skimage.measure import regionprops


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
        return pickle.load(open(address, 'rb'))

    def ef_estimation(self, ed_frame, es_frame, model):
        ed_rp = self._frame_to_rp(ed_frame)
        es_rp = self._frame_to_rp(es_frame)
        ed_vol = model.predict(ed_rp)
        es_vol = model.predict(es_rp)
        return (ed_vol - es_vol) / ed_vol * 100

    def _frame_to_rp(self, frame):
        rp = []
        rp.append(regionprops(frame.astype(np.int64))[0].area)
        rp.append(regionprops(frame.astype(np.int64))[0].convex_area)
        rp.append(regionprops(frame.astype(np.int64))[0].eccentricity)
        rp.append(regionprops(frame.astype(np.int64))[0].major_axis_length)
        rp.append(regionprops(frame.astype(np.int64))[0].minor_axis_length)
        rp.append(regionprops(frame.astype(np.int64))[0].orientation)
        rp = np.array(rp).reshape(-1, 1)
        return rp

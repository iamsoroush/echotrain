import numpy as np
import pickle
from skimage.measure import regionprops


class EFEstimation:

    def __init__(self):
        pass

    @staticmethod
    def load_model(address):
        return pickle.load(open(address, 'rb'))

    def volume_estimation(self,frame,model):
        frame_rp = self.frame_to_rp(frame)
        volume = model.predict(frame_rp)
        return volume

    def ef_estimation(self, ed_frame, es_frame, model):
        ed_rp = self.frame_to_rp(ed_frame)
        es_rp = self.frame_to_rp(es_frame)
        ed_vol = model.predict(ed_rp)
        es_vol = model.predict(es_rp)
        return (ed_vol - es_vol) / ed_vol * 100

    @staticmethod
    def frame_to_rp(frame):
        rp = []
        rp.append(regionprops(frame.astype(np.int64))[0].area)
        rp.append(regionprops(frame.astype(np.int64))[0].convex_area)
        rp.append(regionprops(frame.astype(np.int64))[0].eccentricity)
        rp.append(regionprops(frame.astype(np.int64))[0].major_axis_length)
        rp.append(regionprops(frame.astype(np.int64))[0].minor_axis_length)
        rp.append(regionprops(frame.astype(np.int64))[0].orientation)
        rp = np.array(rp).reshape(1,-1)
        return rp

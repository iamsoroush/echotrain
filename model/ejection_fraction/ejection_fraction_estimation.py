import numpy as np
import pickle
from skimage.measure import regionprops


class EFEstimation:
    """
    This class estimates the volume of a frame or the ejection fraction(ef)
    from end_systolic and end_diastolic label frames.

    HOW TO:
    ef_es = EFEstimation()
        #calling the class
    rp_of_frame = ef_es.frame_to_rp(frame)
        #returns region_properties (rp) of the frame
    ef = ef_es.ef_estimation(ed_frame, es_frame, model)
        #returns ef from ed_frame,es_frame and model for rp to volume transformation
    volume_of_frame = ef_es.volume_estimation(frame,model)
        #returns volume of left ventricle of a frame with the model for rp to volume transformation
    model = ef_es.load_model(address)
        #returns load model in the address directory
    """

    def __init__(self):
        pass

    @staticmethod
    def load_model(address):
        """

        Args:
            address:the directory that the model is saved.

        Returns:
            model saved in address directory
        """
        return pickle.load(open(address, 'rb'))

    def volume_estimation(self,frame,model):
        """

        Args:
            frame: numpy frame of label
            model: model for rp to volume transformation

        Returns:
            volume of left_ventricle of the label frame
        """
        frame_rp = self.frame_to_rp(frame)
        volume = model.predict(frame_rp)
        return volume

    def ef_estimation(self, ed_frame, es_frame, model):
        """

        Args:
            ed_frame: end_diastolic frame in numpy format
            es_frame: end_systolic frame in numpy format
            model: model for rp to volume transformation

        Returns:
            ejection_fraction in percentage

        """
        ed_rp = self.frame_to_rp(ed_frame)
        es_rp = self.frame_to_rp(es_frame)
        ed_vol = model.predict(ed_rp)
        es_vol = model.predict(es_rp)
        return (ed_vol - es_vol) / ed_vol * 100

    @staticmethod
    def frame_to_rp(frame):
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
        rp = []
        rp.append(regionprops(frame.astype(np.int64))[0].area)
        rp.append(regionprops(frame.astype(np.int64))[0].convex_area)
        rp.append(regionprops(frame.astype(np.int64))[0].eccentricity)
        rp.append(regionprops(frame.astype(np.int64))[0].major_axis_length)
        rp.append(regionprops(frame.astype(np.int64))[0].minor_axis_length)
        rp.append(regionprops(frame.astype(np.int64))[0].orientation)
        rp = np.array(rp).reshape(1,-1)
        return rp

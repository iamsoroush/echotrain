from model.ejection_fraction.ejection_fraction_base import EFBase
from tensorflow.keras.models import load_model
from dataset.dataset_ef import EFDataset
import os
import shutil
from skimage.measure import regionprops
import pandas as pd
import skimage.io as io
import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
# % matplotlib
# inline

class EFModel_Simpson(EFBase):

    def __init__(self, config):

        super().__init__(config)
        self.config = config
        self._get_config()

    def simpson_volume(self, frame):
        """

        Args:
            frame: the segmentation map

        Returns:
            the volume
        """
        pi = math.pi
        interval = 5
        ef_pred = []
        volumes = []
        label_frame = frame.astype(np.int64).reshape(112, 112)
        for i in range(0, len(label_frame), interval):
            volume = (label_frame[i].sum() / 2) ** 2 * pi
        return volume
        # for patient in ed_es:
        #     areas = []
        #     for ed_es_frames in patient:
        #         area = 0
        #         for i in range(0, len(ed_es_frames), interval):
        #             area += (ed_es_frames[i].sum() / 2) ** 2 * pi
        #         areas.append(area)
            # ef_pred.append((areas[0] - areas[1]) / areas[0])
        # rp = [regionprops(label_frame)[0].area,
        #       regionprops(label_frame)[0].major_axis_length,
        #       ]
        # area = rp[0]
        # length = rp[1]
        # vol = 0.85 * (area * area) / length
        # return vol

    def ef_estimation(self, ed_frame, es_frame):
        ed_vol = self.simpson_volume(ed_frame)
        es_vol = self.simpson_volume(es_frame)
        return float((ed_vol - es_vol) / ed_vol * 100)

    def train(self):
        pass

    def export(self, model):
        pass

    def _load_simpson_model(self):
        pass

    def _load_simpson_to_vol_model(self):
        pass

    def _get_config(self):
        """

        Get needed information from config.yaml file

        """
        try:
            self.estimation_method = self.config.estimation_method
            if self.estimation_method == "simpson":
                self.al_address = self.config.al.al_address
                # self.al_to_vol_model_dir = self.config.al.al_to_vol_model_dir
                # self.al_to_vol_model_type = self.config.al.train.model_type
                # self.exported_dir = self.config.exported_dir
            else:
                raise ValueError
        except ValueError:
            print("estimation_method should be 'simpson'.")

    def _simpson_to_vol_model(self):
        pass

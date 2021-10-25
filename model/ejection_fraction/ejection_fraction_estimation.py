import cv2
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from echotrain.dataset.dataset_camus import CAMUSDataset
from echotrain.dataset.dataset_generator import DatasetGenerator


class EFEstimation:

    def __init__(self, config):

        self.dataset_class = config.dataset_class
        self.batch_size = config.data_handler.batch_size
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels = config.n_channels

    def ef_estimation(self, ed_frame, es_frame):

        area_train, volume_train = self._data_for_training()
        model = self._area_to_volume_model(area_train, volume_train)

        ed_volume = self._area_to_volume_conversion(self._area(ed_frame), model)
        es_volume = self._area_to_volume_conversion(self._area(es_frame), model)

        return (ed_volume - es_volume) / ed_volume

    @staticmethod
    def _area_to_volume_conversion(area, model):
        return model.predict(area)

    @staticmethod
    def _area_to_volume_model(area_train, volume_train):

        gbr = GradientBoostingRegressor(n_estimators=50)
        gbr.fit(area_train, volume_train)
        return gbr

    @staticmethod
    def _area(image):
        return float(cv2.countNonZero(image))

    def _data_for_training(self):

        if self.dataset_class == 'dataset.dataset_camus.CAMUSDataset':
            camus = CAMUSDataset(config)
            DF = camus.train_df
            dictdir = {}
            for i in DF.index:
                dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[0]
            gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                   , self.batch_size, (self.input_h, self.input_w), self.n_channels)
            list_of_labels = gen.generate_y(dictdir)
            area_list = []
            for frame in list_of_labels:
                area_list.append(self._area(frame))
            volume_list = []
            for i in DF.index:
                if DF.loc[i, ['stage']].astype('string')[0] == 'ED':
                    volume_list.append(DF.loc[i, ['lv_edv']].astype('float32')[0])
                elif DF.loc[i, ['stage']].astype('string')[0] == 'ES':
                    volume_list.append(DF.loc[i, ['lv_esv']].astype('float32')[0])
            area_list = np.array(area_list).reshape(-1, 1)
            volume_list = np.array(volume_list).reshape(-1, 1)

            return area_list, volume_list

        if self.dataset_class == 'dataset.dataset_echonet.EchoNetDataset':
            echonet = EchoNetDataset(config)
            DF = echonet.train_df
            dictdir = {}
            for i in DF.index:
                dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[0]
            gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                   , self.batch_size, (self.input_h, self.input_w), self.n_channels)
            list_of_labels = gen.generate_y(dictdir)
            area_list = []
            for frame in list_of_labels:
                area_list.append(self._area(frame))
            volume_list = []
            for i in DF.index:
                if DF.loc[i, ['stage']].astype('string')[0] == 'ED':
                    volume_list.append(DF.loc[i, ['lv_edv']].astype('float32')[0])
                elif DF.loc[i, ['stage']].astype('string')[0] == 'ES':
                    volume_list.append(DF.loc[i, ['lv_esv']].astype('float32')[0])
            area_list = np.array(area_list).reshape(-1, 1)
            volume_list = np.array(volume_list).reshape(-1, 1)

            return area_list, volume_list

from echotrain.model.metric import mae, mse, r2_score
from echotrain.model.ejection_fraction.ejection_fraction_estimation import EFEstimation
from echotrain.dataset.dataset_camus import CAMUSDataset
from echotrain.dataset.dataset_generator import DatasetGenerator
from echotrain.dataset.dataset_echonet import EchoNetDataset
from skimage.measure import regionprops
import numpy as np
import cv2
import pickle

class EFEvaluation:

    def __init__(self,config):

        self.config = config
        self.dataset_class = config.dataset_class
        self.batch_size = config.data_handler.batch_size
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels = config.n_channels

    def train_AtoV_model(self,model):

        area_train, volume_train = self.data_for_training_FtoV()
        model = model.fit(area_train, volume_train)
        return model

    @staticmethod
    def save_model(model ,name): #.sav format
        pickle.dump(model,open(name ,'wb'))

    def evaluation_of_ef_model(self, model):

        efe = EFEstimation(self.config)
        ed_es_data = self._data_for_ef_evaluation('val')[0]
        ef_true = self._data_for_ef_evaluation('val')[1]
        ef_pred = []
        for i in range(len(ed_es_data)):
            ef_pred.append(efe.ef_estimation(ed_es_data[i][0], ed_es_data[i][1], model))
        ef_pred = np.array(ef_pred)
        return {'mean_absoulute_error_validation' : mae(ef_true, ef_pred),
                'mean_squared_error_validation' : mse(ef_true, ef_pred)}
                #'r2-score_validation' : r2_score(ef_true, ef_pred)}

    def data_for_training_RPtoV(self,dataset_type='train'):

        if dataset_type == 'train':
            frames, volumes = self.data_for_training_FtoV()
            rps=[]
            for frame in frames:
                rp=[]
                rp.append(regionprops(frame.astype(np.int64))[0].area)
                rp.append(regionprops(frame.astype(np.int64))[0].convex_area)
                rp.append(regionprops(frame.astype(np.int64))[0].eccentricity)
                rp.append(regionprops(frame.astype(np.int64))[0].major_axis_length)
                rp.append(regionprops(frame.astype(np.int64))[0].minor_axis_length)
                rp.append(regionprops(frame.astype(np.int64))[0].orientation)
                rps.append(rp)
            rps=np.array(rps)
            return rps , volumes
        elif dataset_type == 'val':
            frames, volumes = self.data_for_validating_FtoV()
            rps = []
            for frame in frames:
                rp = []
                rp.append(regionprops(frame.astype(np.int64))[0].area)
                rp.append(regionprops(frame.astype(np.int64))[0].convex_area)
                rp.append(regionprops(frame.astype(np.int64))[0].eccentricity)
                rp.append(regionprops(frame.astype(np.int64))[0].major_axis_length)
                rp.append(regionprops(frame.astype(np.int64))[0].minor_axis_length)
                rp.append(regionprops(frame.astype(np.int64))[0].orientation)
                rps.append(rp)
            rps = np.array(rps)
            return rps, volumes

    def data_for_training_FtoV(self):

        if self.dataset_class == 'dataset.dataset_camus.CAMUSDataset':
            camus = CAMUSDataset(self.config)
            DF = camus.train_df
            dictdir = {}
            for i in DF.index:
                dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[0]
            gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                   , self.batch_size, (self.input_h, self.input_w), self.n_channels)
            frame_list = gen.generate_y(dictdir)
            volume_list = []
            for i in DF.index:
                if DF.loc[i, ['stage']].astype('string')[0] == 'ED':
                    volume_list.append(DF.loc[i, ['lv_edv']].astype('float32')[0])
                elif DF.loc[i, ['stage']].astype('string')[0] == 'ES':
                    volume_list.append(DF.loc[i, ['lv_esv']].astype('float32')[0])

            volume_list = np.array(volume_list).reshape(-1, 1)

            return frame_list, volume_list

        if self.dataset_class == 'dataset.dataset_echonet.EchoNetDataset':
            echonet = EchoNetDataset(self.config)
            DF = echonet.train_df
            dictdir = {}
            for i in DF.index:
                dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[0]
            gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                   , self.batch_size, (self.input_h, self.input_w), self.n_channels)
            frame_list = gen.generate_y(dictdir)
            volume_list = []
            for i in DF.index:
                if DF.loc[i, ['stage']].astype('string')[0] == 'ED':
                    volume_list.append(DF.loc[i, ['lv_edv']].astype('float32')[0])
                elif DF.loc[i, ['stage']].astype('string')[0] == 'ES':
                    volume_list.append(DF.loc[i, ['lv_esv']].astype('float32')[0])
            volume_list = np.array(volume_list).reshape(-1, 1)

            return frame_list, volume_list

    def data_for_validating_FtoV(self):

        if self.dataset_class == 'dataset.dataset_camus.CAMUSDataset':
            camus = CAMUSDataset(self.config)
            DF = camus.val_df_
            dictdir = {}
            for i in DF.index:
                dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[0]
            gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                   , self.batch_size, (self.input_h, self.input_w), self.n_channels)
            frame_list = gen.generate_y(dictdir)
            volume_list = []
            for i in DF.index:
                if DF.loc[i, ['stage']].astype('string')[0] == 'ED':
                    volume_list.append(DF.loc[i, ['lv_edv']].astype('float32')[0])
                elif DF.loc[i, ['stage']].astype('string')[0] == 'ES':
                    volume_list.append(DF.loc[i, ['lv_esv']].astype('float32')[0])

            volume_list = np.array(volume_list).reshape(-1, 1)

            return frame_list, volume_list

        if self.dataset_class == 'dataset.dataset_echonet.EchoNetDataset':
            echonet = EchoNetDataset(self.config)
            DF = echonet.train_df
            dictdir = {}
            for i in DF.index:
                dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[0]
            gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                   , self.batch_size, (self.input_h, self.input_w), self.n_channels)
            frame_list = gen.generate_y(dictdir)
            volume_list = []
            for i in DF.index:
                if DF.loc[i, ['stage']].astype('string')[0] == 'ED':
                    volume_list.append(DF.loc[i, ['lv_edv']].astype('float32')[0])
                elif DF.loc[i, ['stage']].astype('string')[0] == 'ES':
                    volume_list.append(DF.loc[i, ['lv_esv']].astype('float32')[0])
            volume_list = np.array(volume_list).reshape(-1, 1)

            return frame_list, volume_list

    def _data_for_ef_evaluation(self, dataset_type):

        if dataset_type == 'train':
            if self.dataset_class == 'dataset.dataset_echonet.EchoNetDataset':
                echonet = EchoNetDataset(self.config)
                DF = echonet.train_df
                dictdir = {}
                for i in DF.index:
                    dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[
                        0]
                gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                       , self.batch_size, (self.input_h, self.input_w), self.n_channels)
                list_of_labels = gen.generate_y(dictdir)
                ef_label = []
                es_ev_patients = []
                for i in range(len(DF['case_id'].unique())):
                    ef_label.append(DF.loc[DF['case_id'] == DF['case_id'].unique()[0], :].loc[DF['stage'] == 'ES', ['lv_ef']]['lv_ef'].astype('float'))

                    es_ev_each_patient = []
                    es_ev_each_patient.append(list_of_labels[DF.index.get_loc(
                        DF.loc[DF['case_id'] == DF['case_id'].unique()[0],:][DF['stage'] == 'ED'].index[0])])
                    es_ev_each_patient.append(list_of_labels[DF.index.get_loc(
                        DF.loc[DF['case_id'] == DF['case_id'].unique()[0], :][DF['stage'] == 'ES'].index[0])])
                    es_ev_patients.append(es_ev_each_patient)
                return np.array(es_ev_patients),np.array(ef_label)

        if dataset_type == 'test':
            if self.dataset_class == 'dataset.dataset_echonet.EchoNetDataset':
                echonet = EchoNetDataset(self.config)
                DF = echonet.test_df
                dictdir = {}
                for i in DF.index:
                    dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[
                        0]
                gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                       , self.batch_size, (self.input_h, self.input_w), self.n_channels)
                list_of_labels = gen.generate_y(dictdir)
                ef_label = []
                es_ev_patients = []
                for i in range(len(DF['case_id'].unique())):
                    ef_label.append(DF.loc[DF['case_id'] == DF['case_id'].unique()[0], :].loc[DF['stage'] == 'ES', ['lv_ef']]['lv_ef'].astype('float'))

                    es_ev_each_patient = []
                    es_ev_each_patient.append(list_of_labels[DF.index.get_loc(
                        DF.loc[DF['case_id'] == DF['case_id'].unique()[0],:][DF['stage'] == 'ED'].index[0])])
                    es_ev_each_patient.append(list_of_labels[DF.index.get_loc(
                        DF.loc[DF['case_id'] == DF['case_id'].unique()[0], :][DF['stage'] == 'ES'].index[0])])
                    es_ev_patients.append(es_ev_each_patient)
                return np.array(es_ev_patients),np.array(ef_label)

        if dataset_type == 'val':
            if self.dataset_class == 'dataset.dataset_echonet.EchoNetDataset':
                echonet = EchoNetDataset(self.config)
                DF = echonet.val_df_
                dictdir = {}
                for i in DF.index:
                    dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[
                        0]
                gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                                       , self.batch_size, (self.input_h, self.input_w), self.n_channels)
                list_of_labels = gen.generate_y(dictdir)
                ef_label = []
                es_ev_patients = []
                for i in range(len(DF['case_id'].unique())):
                    ef_label.append(DF.loc[DF['case_id'] == DF['case_id'].unique()[0], :].loc[DF['stage'] == 'ES', ['lv_ef']]['lv_ef'].astype('float'))

                    es_ev_each_patient = []
                    es_ev_each_patient.append(list_of_labels[DF.index.get_loc(
                        DF.loc[DF['case_id'] == DF['case_id'].unique()[0],:][DF['stage'] == 'ED'].index[0])])
                    es_ev_each_patient.append(list_of_labels[DF.index.get_loc(
                        DF.loc[DF['case_id'] == DF['case_id'].unique()[0], :][DF['stage'] == 'ES'].index[0])])
                    es_ev_patients.append(es_ev_each_patient)
                return np.array(es_ev_patients),np.array(ef_label)
    @staticmethod
    def area(image):
        return float(cv2.countNonZero(image))
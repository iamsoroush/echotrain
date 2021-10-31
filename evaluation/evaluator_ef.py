from echotrain.model.metric import mae, mse, r2_score
from echotrain.model.ejection_fraction.ejection_fraction_estimation import EFEstimation
from echotrain.dataset.dataset_camus import CAMUSDataset
from echotrain.dataset.dataset_generator import DatasetGenerator
from echotrain.dataset.dataset_echonet import EchoNetDataset
import numpy as np
import pickle


class EFEvaluation:

    def __init__(self, config):

        self.config = config
        self.dataset_class = config.dataset_class
        self.batch_size = config.data_handler.batch_size
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels = config.n_channels

    @staticmethod
    def save_model(model, name):  # .sav format
        pickle.dump(model, open(name, 'wb'))

    def evaluation_of_ef_model(self, model):

        efe = EFEstimation()
        ed_es_data = self._data_for_ef_evaluation('val')[0]
        ef_true = self._data_for_ef_evaluation('val')[1]
        ef_pred = []
        for i in range(len(ed_es_data)):
            ef_pred.append(efe.ef_estimation(ed_es_data[i][0], ed_es_data[i][1], model))
        ef_pred = np.array(ef_pred)
        return {'mean_absoulute_error_validation': mae(ef_true, ef_pred),
                'mean_squared_error_validation': mse(ef_true, ef_pred)}
        # 'r2-score_validation' : r2_score(ef_true, ef_pred)}

    def data_for_rptov(self,dataset_type):

        efe = EFEstimation()
        frames, volumes = self.data_for_ftov(dataset_type)
        rps = []
        for frame in frames:
            rp = []
            rp.append(efe.frame_to_rp(frame))
            rps.append(rp)
        return np.array(rps).reshape(-1, 6)

    def data_for_ftov(self, dataset_type):

        if self.dataset_class == 'dataset.dataset_camus.CAMUSDataset':
            camus = CAMUSDataset(self.config)
            if dataset_type == 'train':
                DF = camus.train_df
            if dataset_type == 'val':
                DF = camus.val_df_
            if dataset_type == 'test':
                DF = camus.test_df_
        elif self.dataset_class == 'dataset.dataset_echonet.EchoNetDataset':
            echonet = EchoNetDataset(self.config)
            if dataset_type == 'train':
                DF = echonet.train_df
            if dataset_type == 'val':
                DF = echonet.val_df_
            if dataset_type == 'test':
                DF = echonet.test_df_

        image_paths = {}
        volumes = []
        for i in DF.index:
            if DF.loc[i, ['stage']].astype('string')[0] == 'ED':
                image_paths[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[
                    0]
                volumes.append(DF.loc[i, ['lv_edv']].astype('float')[0])
            if DF.loc[i, ['stage']].astype('string')[0] == 'ES':
                image_paths[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[
                    0]
                volumes.append(DF.loc[i, ['lv_esv']].astype('float')[0])
        gen = DatasetGenerator(np.array(list(image_paths.keys())), image_paths, self.batch_size
                               , (self.input_h, self.input_w), self.n_channels)

        frames = np.array(gen.generate_y(image_paths))
        volumes = np.array(volumes)
        return frames, volumes

    def _data_for_ef_evaluation(self, dataset_type):

        echonet = EchoNetDataset(self.config)
        if dataset_type == 'train':
            DF = echonet.train_df
        elif dataset_type == 'test':
            DF = echonet.test_df_
        elif dataset_type == 'val':
            DF = echonet.val_df_

        dictdir = {}
        for i in DF.index:
            dictdir[DF.loc[i, ['image_path']].astype('string')[0]] = DF.loc[i, ['label_path']].astype('string')[
                0]
        gen = DatasetGenerator(list(DF['image_path'].astype('string')), dictdir
                               , self.batch_size, (self.input_h, self.input_w), self.n_channels)
        ef_list = []
        ed_es_list = []
        for case in DF['case_id'].unique():
            es_x_path = DF[DF['case_id'] == case][DF['stage'] == 'ES']['image_path'].astype('string')
            ed_x_path = DF[DF['case_id'] == case][DF['stage'] == 'ED']['image_path'].astype('string')
            es_y_path = DF[DF['case_id'] == case][DF['stage'] == 'ES']['label_path'].astype('string')
            ed_y_path = DF[DF['case_id'] == case][DF['stage'] == 'ED']['label_path'].astype('string')
            ef = DF[DF['case_id'] == case][DF['stage'] == 'ED']['lv_ef'].astype('float')
            ed_es_frames = gen.generate_y({ed_x_path[ed_x_path.index[0]]: ed_y_path[ed_y_path.index[0]],
                                           es_x_path[es_x_path.index[0]]: es_y_path[es_y_path.index[0]]})
            ef_list.append(ef[ef.index[0]])
            ed_es_list.append(ed_es_frames)
        ef_list = np.array(ef_list)
        ed_es_list = np.array(ed_es_list)
        return ed_es_list, ef_list

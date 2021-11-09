from model.metric import mae, mse, r2_score
from dataset.dataset_ef import EFDataset
from dataset.dataset_echonet import EchoNetDataset
from pydoc import locate
import pandas as pd
import numpy as np
import os


class EFEvaluator:

    def __init__(self, config):

        self.config = config
        self._get_config()

    def evaluate(self, dataset_subset):

        ef_dataset = EFDataset(self.config)
        ed_es_frames, ef_true = ef_dataset.ef_dataset(self.dataset_type, dataset_subset)
        df_ef = {'ef_true': [],
                 'ef_pred': [],
                 'mae': [],
                 'mse': [],
                 'r2_score': []}
        ef_pred = []

        for ed_es_frame in ed_es_frames:
            ef_pred.append(self.model_class.ef_estimation(ed_es_frame[0].reshape(1, 112, 112, 1)
                                                          , ed_es_frame[1].reshape(1, 112, 112, 1)))

        ef_pred = np.array(ef_pred).reshape(-1, 1)
        ef_true = ef_true.reshape(-1, 1)
        for i in range(len(ef_true)):
            df_ef['ef_true'].append(ef_true[i])
            df_ef['ef_pred'].append(ef_pred[i])
            df_ef['mae'].append(mae(ef_true[i], ef_pred[i]))
            df_ef['mse'].append(mse(ef_true[i], ef_pred[i]))
            df_ef['r2_score'].append(r2_score(ef_true[i], ef_pred[i]))

        echonet = EchoNetDataset(self.config)
        dataset_df = echonet.val_df_
        dataset_df.reset_index(inplace=True)
        dataset_df['ef_true'] = None
        dataset_df['ef_pred'] = None
        dataset_df['mae'] = None
        dataset_df['mse'] = None
        df_ef = pd.DataFrame(df_ef)
        for i in range(len(dataset_df['case_id'].unique())):
            dataset_df.loc[i * 2, ['ef_true', 'ef_pred', 'mae', 'mse']] = df_ef.loc[
                i, ['ef_true', 'ef_pred', 'mae', 'mse']]
            dataset_df.loc[i * 2 + 1, ['ef_true', 'ef_pred', 'mae', 'mse']] = df_ef.loc[
                i, ['ef_true', 'ef_pred', 'mae', 'mse']]
        dataset_df.to_csv(os.path.join(self.exported_dir, 'exported', 'dataframe_of_evaluation.csv'))
        return dataset_df

    def _get_config(self):

        self.estimation_method = self.config.estimation_method
        if self.estimation_method == 'encoder':
            self.dataset_type = 'image'
            model_class_location = locate('model.ejection_fraction.ef_model_with_encoder.EFModel_Encoder')
            self.model_class = model_class_location(self.config)
            self.exported_dir = self.config.exported_dir
        elif self.estimation_method == 'rp':
            self.dataset_type = 'label'
            model_class_location = locate('model.ejection_fraction.ef_model_with_rp.EFModel_RP')
            self.model_class = model_class_location(self.config)
            self.exported_dir = self.config.exported_dir

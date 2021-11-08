from model.metric import mae, mse, r2_score
from dataset.dataset_ef import EFDataset
from pydoc import locate
import pandas as pd

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
        for i in range(len(ed_es_frames)):
            df_ef['ef_pred'].append(self.model_class.ef_estimation(ed_es_frames[i][0], ed_es_frames[i][1]))
            df_ef['ef_true'].append(ef_true[i])
            df_ef['mae'].append(mae(ef_true[i], self.model_class.ef_estimation(ed_es_frames[i][0], ed_es_frames[i][1])))
            df_ef['mse'].append(mse(ef_true[i], self.model_class.ef_estimation(ed_es_frames[i][0], ed_es_frames[i][1])))
            df_ef['r2_score'].append(
                r2_score(ef_true[i], self.model_class.ef_estimation(ed_es_frames[i][0], ed_es_frames[i][1])))

        return pd.DataFrame(df_ef)

    def _get_config(self):

        self.estimation_method = self.config.estimation_method
        if self.estimation_method == 'encoder':
            self.dataset_type = 'image'
            model_class_location = locate('model.ejection_fraction.ef_model_with_encoder.EFModel_Encoder')
            self.model_class = model_class_location(self.config)
        elif self.estimation_method == 'rp':
            self.dataset_type = 'label'
            model_class_location = locate('model.ejection_fraction.ef_model_with_rp.EFModel_RP')
            self.model_class = model_class_location(self.config)

import sys
sys.path.append('..')

from .evaluator import Evaluator
from dataset.dataset_echonet import EchoNetDataset


class EvaluatorEchoNet(Evaluator):

    def __init__(self, exported_dir):
        super().__init__(exported_dir)

    def generate_report(self):

        """Generates report using given config file and exported model"""

        test_data_gen, n_iter_test, test_df = self._create_test_data_gen()
        preprocessed_data_gen = self._add_preprocessing(test_data_gen)
        inference_model = self._load_model()

        eval_report = self.build_data_frame(inference_model, preprocessed_data_gen, n_iter_test, test_df.index)
        return eval_report, test_df

    def _create_test_data_gen(self):
        print('preparing dataset ...')
        dataset = EchoNetDataset(config=None)
        _, test_data_gen, _, n_iter_test = dataset.create_test_data_generator()

        return test_data_gen, n_iter_test, dataset.test_df

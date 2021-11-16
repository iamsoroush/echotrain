from echotrain.evaluation.evaluator import Evaluator
from echotrain.dataset.dataset_echonet import EchoNetDataset


class EvaluatorEchoNet(Evaluator):

    def __init__(self, exported_dir):
        super().__init__(exported_dir)

    def generate_report(self, on='test'):

        """Generates report using given config file and exported model"""

        assert on in ('validation', 'test'), 'pass either "test" or "validation" for "on" argument.'

        if on is 'test':
            data_gen, n_iter, df = self._create_test_data_gen()
        else:
            data_gen, n_iter, df = self._create_val_data_gen()

        preprocessed_data_gen = self._add_preprocessing(data_gen)
        inference_model = self._load_model()

        eval_report = self.build_data_frame(inference_model, preprocessed_data_gen, n_iter, df.index)

        return eval_report, df

    def _create_test_data_gen(self):
        print('preparing test dataset ...')
        dataset = EchoNetDataset(config=None)
        test_data_gen, n_iter_test = dataset.create_test_data_generator()

        return test_data_gen, n_iter_test, dataset.test_df

    def _create_val_data_gen(self):
        print('preparing validation dataset ...')
        dataset = EchoNetDataset(config=None)
        val_data_gen, n_iter_val = dataset.create_validation_data_generator()

        return val_data_gen, n_iter_val, dataset.validation_df


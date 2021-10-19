from abc import ABC, abstractmethod
from pydoc import locate
import pathlib

from utils.handling_yaml import load_config_file


class EvaluatorBase(ABC):

    def __init__(self, exported_dir):
        self.exported_dir = exported_dir
        self.config = None
        self._load_config()

    @abstractmethod
    def build_data_frame(self, model, data_gen_val_preprocessed, n_iter, val_data_indexes):

        """Generates a report as a pandas dataframe, each row represents a single data point (evaluation image)."""

        pass

    def generate_report(self):

        """Generates report using given config file and exported model"""

        val_data_gen, n_iter_val, val_df = self._create_val_data_gen()
        inference_model = self._load_model()

        eval_report = self.build_data_frame(inference_model, val_data_gen, n_iter_val, val_df.index)
        return eval_report, val_df

    def _load_config(self):
        if self.exported_dir is not None:
            config_path = list(pathlib.Path(self.exported_dir).glob('*.yaml'))[0]
            self.config = load_config_file(config_path)

    def _create_val_data_gen(self):

        dataset_class_path = self.config.dataset_class
        preprocessor_class_path = self.config.preprocessor_class

        # Dataset
        print('preparing dataset ...')
        dataset_class = locate(f'{dataset_class_path}')
        dataset = dataset_class(self.config)
        _, val_data_gen, _, n_iter_val = dataset.create_data_generators()

        # Preprocessor
        print('preparing pre-processor ...')
        preprocessor_class = locate(f'{preprocessor_class_path}')
        preprocessor = preprocessor_class(self.config)
        val_data_gen = preprocessor.add_preprocess(val_data_gen, False)
        return val_data_gen, n_iter_val, dataset.validation_df

    def _load_model(self):

        """Loads and returns the ``tf.keras.Model`` based on config file and .hdf5 file

        :param exported_dir: path to exported folder
        :returns model: ``tf.keras.Model`` ready to make predictions

        :raises AssertionError: could not import model_class
        :raises Exception: f'could not find a checkpoint(.hdf5) file on {base_dir}'
        """

        # Model
        print('preparing model ...')
        model_class_path = self.config.model_class
        model_class = locate(f'{model_class_path}')
        model_obj = model_class(config=self.config)

        try:
            checkpoint_path = list(pathlib.Path(self.exported_dir).glob('*.hdf5'))[0]
        except IndexError:
            raise Exception(f'could not find a checkpoint(.hdf5) file on {self.exported_dir}')

        model = model_obj.load_model(checkpoint_path=checkpoint_path)
        print(f'loaded model using this checkpoint: {checkpoint_path}')
        return model

from abc import ABC, abstractmethod
from pydoc import locate
import pathlib


class EvaluatorBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def build_data_frame(self, model, data_gen_val_preprocessed, n_iter, val_data_indexes):

        """Generates a report as a pandas dataframe, each row represents a single data point (evaluation image)."""

        pass

    @abstractmethod
    def generate_report(self, exported_dir):

        """Generates report using given config file and exported model"""

        pass

    @staticmethod
    def _create_val_data_gen(config):

        dataset_class_path = config.dataset_class
        preprocessor_class_path = config.preprocessor_class

        # Dataset
        print('preparing dataset ...')
        dataset_class = locate(f'{dataset_class_path}')
        dataset = dataset_class(config)
        _, val_data_gen, _, n_iter_val = dataset.create_data_generators()

        # Preprocessor
        print('preparing pre-processor ...')
        preprocessor_class = locate(f'{preprocessor_class_path}')
        preprocessor = preprocessor_class(config)
        val_data_gen = preprocessor.add_preprocess(val_data_gen, False)
        return val_data_gen, n_iter_val, dataset.validation_df

    @staticmethod
    def _load_model(config, exported_dir):

        """Loads and returns the ``tf.keras.Model`` based on config file and .hdf5 file

        :param exported_dir: path to exported folder
        :returns model: ``tf.keras.Model`` ready to make predictions

        :raises AssertionError: could not import model_class
        :raises Exception: f'could not find a checkpoint(.hdf5) file on {base_dir}'
        """

        # Model
        print('preparing model ...')
        model_class_path = config.model_class
        model_class = locate(f'{model_class_path}')
        model_obj = model_class(config=config)

        try:
            checkpoint_path = list(pathlib.Path(exported_dir).glob('*.hdf5'))[0]
        except IndexError:
            raise Exception(f'could not find a checkpoint(.hdf5) file on {exported_dir}')

        model = model_obj.load_model(checkpoint_path=checkpoint_path)
        print(f'loaded model using this checkpoint: {checkpoint_path}')
        return model

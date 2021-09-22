from pathlib import Path
from pydoc import locate

import yaml
import numpy as np


class EchoInference:

    def __init__(self, exported_dir):

        """Checks config and load model and preprocessor

        :param exported_dir: path to exported folder containing config and checkpoint files

        Attributes:
            config: config file as a python object
            model_class: model's class name to import
            preprocessor_class: preprocessor's class name to import
            model_obj: model's class, contains .load_model() and .post_process() for single image
            model: tf.keras.Model ready to predict
            preprocessor: preprocessor object, ready to preprocess the input image for model.predict
        """

        self.config, self.model_class, self.preprocessor_class = None, None, None
        self._check_config_file(exported_dir)

        self.model_obj, self.model = self._load_model(exported_dir)
        self.preprocessor = self._load_preprocessor()

    def pre_process(self, image):

        """Preprocesses input image in order to make predictions, same as training-time pre-processing.

        :param image: rgb(0, 255) image

        :returns preprocessed_image: this image is ready for processing
        """

        return self.preprocessor.pre_process(image)

    def process(self, pre_processed_image):

        """Processes pre-processed image using the trained model

        :param pre_processed_image: use self.pre_process to pre-process your image

        :returns processing_output: raw processed output.
        """

        return self.model.predict(np.expand_dims(pre_processed_image, 0))[0]

    def post_process(self, processed_image):

        """Postprocesses the raw output of processing step. This method returns the final result.

        :param processed_image: use self.process method's output

        :returns final_result: ready-to-go result
        """

        raise self.model_obj.post_process(processed_image)

    def _check_config_file(self, base_dir):

        """Checks and loads config file

        :param base_dir: path to exported directory which contains .yaml and .hdf5 files

        :raises Exception(f'could not find any .yaml files on specified path: {base_dir}')
        :raises Exception('could not find model_class') if model_class is not provided in config file
        :raises Exception('could not find preprocessor_class') if preprocessor_class is not provided in config file
        """

        try:
            config_path = list(Path(base_dir).glob('*.yaml'))[0]
        except IndexError:
            raise Exception(f'could not find any .yaml files on specified path: {base_dir}')

        self.config = load_config_file(config_path)

        try:
            self.model_class = self.config.model_class
        except AttributeError:
            raise Exception('could not find model_class')

        try:
            self.preprocessor_class = self.config.preprocessor_class
        except AttributeError:
            raise Exception('could not find preprocessor_class')

    def _load_model(self, base_dir):

        """Loads and returns the tf.keras.Model based on config file and .hdf5 file

        :param base_dir: path to exported folder
        :returns model: tf.keras.Model ready to make predictions

        :raises AssertionError(could not import model_class)
        :raises Exception(f'could not find a checkpoint(.hdf5) file on {base_dir}')
        """

        model_class = locate(self.model_class)
        assert model_class is not None, f'could not import class {self.model_class}'

        try:
            checkpoint_path = list(Path(base_dir).glob('*.hdf5'))[0]
        except IndexError:
            raise Exception(f'could not find a checkpoint(.hdf5) file on {base_dir}')

        model_obj = model_class(config=self.config)
        model = model_obj.load_model(checkpoint_path=checkpoint_path)
        return model_obj, model

    def _load_preprocessor(self):

        """Loads and returns preprocessor based on config file

        :returns preprocessor: PreprocessorBase object to preprocess the input image
        :raises f'could not import class {self.preprocessor_class}'
        """

        preprocessor_class = locate(self.preprocessor_class)
        assert preprocessor_class is not None, f'could not import class {self.preprocessor_class}'

        preprocessor = preprocessor_class(config=self.config)
        return preprocessor


def load_config_file(path):

    """
    loads the json config file and returns a dictionary

    :param path: path to json config file
    :return: a dictionary of {config_name: config_value}
    """

    with open(path) as f:
        # use safe_load instead load
        data_map = yaml.safe_load(f)

    config_obj = Struct(**data_map)
    return config_obj


class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v

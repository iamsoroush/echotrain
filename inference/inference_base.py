import sys
import os
import numpy as np
import tensorflow as tf

# dataset_file_dir = 'D:\AIMedic\FinalProject_echocardiogram\echoC_Codes\source\echotrain\dataset'
# sys.path.insert(0, dataset_dir)
currentdir = os.path.abspath('/echotrain/model')
sys.path.append(currentdir)
from pre_processing import PreProcessor


class InferenceBase:

    def __init__(self, model_dir, config):
        """

        :param model_dir:
        :param config: dictionary of {config_name: config_value}
        """

        self.model_dir = model_dir
        self.config = config
        self.model = None
        self._load_model()

    @staticmethod
    def pre_process(image):
        """Preprocesses input image in order to make predictions, same as training-time pre-processing.

        :param image: rgb(0, 255) image

        :returns preprocessed_image: this image is ready for processing
        """
        preprocessor = PreProcessor()
        pre_processed_image = preprocessor.img_preprocess(image)

        return pre_processed_image

    def process(self, pre_processed_image):
        """Processes pre-processed image using the trained model

        :param pre_processed_image: use self.pre_process to pre-process your image

        :returns processed_output: raw processed output.
        """
        y_prob = self.model.predict(pre_processed_image)
        processed_output = np.argmax(y_prob, axis=2)

        return processed_output

    def post_process(self, processed_image):
        """Postprocesses the raw output of processing step. This method returns the final result.

        :param processed_image: use self.process method's output

        :returns final_result: ready-to-go result
        """
        processed_output = self.process(processed_image)
        # if processed_output.dtype != 'uint':

        # return final_image

    def _load_model(self):
        """Loads the best model and stores as self.model"""

        self.model = tf.keras.models.load_model('../model.h5')

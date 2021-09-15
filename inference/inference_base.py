

class InferenceBase:

    def __init__(self, model_dir, config):

        """

        :param model_dir:
        :param config: dictionary of {config_name: config_value}
        """

        self.model_dir = model_dir
        self.config = config
        self.model = None

    def pre_process(self, image):

        """Preprocesses input image in order to make predictions, same as training-time pre-processing.

        :param image: rgb(0, 255) image

        :returns preprocessed_image: this image is ready for processing
        """

        raise Exception('not implemented')

    def process(self, pre_processed_image):

        """Processes pre-processed image using the trained model

        :param pre_processed_image: use self.pre_process to pre-process your image

        :returns processing_output: raw processed output.
        """

        raise Exception('not implemented')

    def post_process(self, processed_image):

        """Postprocesses the raw output of processing step. This method returns the final result.

        :param processed_image: use self.process method's output

        :returns final_result: ready-to-go result
        """

        raise Exception('not implemented')

    def _load_model(self):

        """Loads the best model and stores as self.model"""

        raise Exception('not implemented')

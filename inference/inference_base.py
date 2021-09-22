import os
from model.base_model import UNet
from model.pre_processing import PreProcessor


class InferenceBase:
    """
    inference base class for getting the model output ready for ui

    HOW TO:

    inference = InferenceBase(model_dir, config)
    pre_processed_image = inference.pre_process(image)
    processed_image = inference.process(pre_processed_image)
    output_image_label = inference.post_process(processed_image)
    """

    def __init__(self, model_dir, config):
        """

        :param model_dir: ?
        :param config: dictionary of {config_name: config_value}
        """

        self.model_dir = model_dir
        self.config = config

        # we consider the self.model_dir as the model directory which contains the "exported" file
        exported_path = os.path.join(self.model_dir, "exported")

        # check if the directory existed
        if not os.path.isdir(exported_path):
            raise Exception('Not a directory')

        # finding the .hdf5 file
        exported_list_dir = os.listdir(exported_path)
        for exp_file_dir in exported_list_dir:
            if exp_file_dir.split('.')[-1] == "hdf5":
                self.checkpoints_dir = exp_file_dir
                break

    def pre_process(self, image):
        """Preprocesses input image in order to make predictions, same as training-time pre-processing.

        :param image: rgb(0, 255) image

        :returns preprocessed_image: this image is ready for processing
        """
        preprocessor = PreProcessor(self.config)
        pre_processed_image = preprocessor.img_preprocess(image)

        return pre_processed_image

    def process(self, pre_processed_image):
        """Processes pre-processed image using the trained model

        :param pre_processed_image: use self.pre_process to pre-process your image

        :returns processed_output: raw processed output.
        """
        model = self._load_model(self.checkpoints_dir)
        y_prob = model.predict(pre_processed_image)
        processed_output = y_prob

        return processed_output

    @staticmethod
    def post_process(processed_image):
        """Postprocesses the raw output of processing step. This method returns the final result.

        :param processed_image: use self.process method's output

        :returns final_result: ready-to-go result
        """
        # post process the output image if needed
        processed_output = processed_image
        if str(processed_output.dtype).find('uint') == -1:
            processed_output = processed_output.astype('uint8')

        return processed_output

    def _load_model(self, checkpoint_path):
        """Loads the best model and stores as self.model from the checkpoint_path

        :param checkpoint_path: checkpoints directory

        :return loaded model
        """

        # retrieve the model architecture
        model = UNet(self.config).get_model_graph()

        # return the pre-trained model
        return model.load_weights(checkpoint_path)

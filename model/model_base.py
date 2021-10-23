from abc import abstractmethod

from .base_class import BaseClass


class ModelBase(BaseClass):

    """Model's abstract class

    Attributes:

        name: model name
        optimizer_type: which optimizer to use?
        learning_rate: initial learning rate of the optimizer
        loss_type: name of the loss function's
        metrics: a list of metrics to use

    """

    @abstractmethod
    def generate_training_model(self):

        """Generates the model for training, and returns the compiled model.

        :returns model: the final model which is ready to be trained.
        """
        pass

    @abstractmethod
    def get_model_graph(self):

        """Defines the model's graph.

        Output of this method will be used for inference (by loading saved checkpoints) and training (by compiling the
        graph)

        :return: a model of type tensorflow.keras.Model
        """
        pass

    def load_model(self, checkpoint_path):

        """Loads the model using given checkpoint

        :param checkpoint_path: path to .hdf5 file
        """

        model = self.get_model_graph()
        model.load_weights(checkpoint_path)

        return model

    @abstractmethod
    def post_process(self, predicted):

        """Post processes the output of self.model.predict

        :param predicted: np.ndarray(input_h, input_w, n_channels).float64, output of the model
        :returns ret: np.ndarray(input_h, input_w, n_channels).int8
        """

        pass

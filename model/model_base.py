from abc import ABC, abstractmethod


class ModelBase(ABC):

    def __init__(self, config):

        """

        :param config: a Python object with attributes as config values

        Attributes
            name str: model name
            optimizer_type str: which optimizer to use?
            learning_rate float: initial learning rate of the optimizer
            loss_type str: loss function's type
            metrics [str]: a list of metrics to use

        """

        # self.name = config.name
        # self.input_h = config.input_h
        # self.input_w = config.input_w
        # self.optimizer_type = config.model.optimizer.type
        # self.learning_rate = config.model.optimizer.initial_lr
        # self.loss_type = config.model.loss_type
        # self.metrics = config.model.metrics
        self.config = config
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels = config.n_channels
        self.inference_threshold = 0.5

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

    def post_process(self, predicted):

        """Post processes the output of self.model.predict

        :param predicted: np.ndarray(input_h, input_w, 1).float64, output of the model
        :returns ret: np.ndarray(input_h, input_w, 1).int8

        """

        return (predicted > self.inference_threshold).astype(int)

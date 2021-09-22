

class ModelBase:

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

    def generate_training_model(self):

        """Generates the model for training, and returns the compiled model.

        :returns model: the final model which is ready to be trained.
        """

        raise Exception('not implemented')

    def get_model_graph(self):

        """Defines the model's graph.

        Output of this method will be used for inference (by loading saved checkpoints) and training (by compiling the
        graph)

        :return: a model of type tensorflow.keras.Model
        """

        raise Exception('not implemented')

    def load_model(self, checkpoint_path):

        """Loads the model using given checkpoint

        :param checkpoint_path: path to .hdf5 file
        """

        raise Exception('not implemented')

    def post_process(self, predicted):

        """Post processes the output of self.model.predict

        :param predicted: output of the model, (input_h, input_w, 1).float64
        :returns ret: np.ndarray of size(input_h, input_w, 1).int8

        """

        raise Exception('not implemented')

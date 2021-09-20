
class BaseModel:

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

        self.name = config.model.name
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.n_channels=config.n_channels
        self.optimizer_type = config.model.optimizer.type
        self.learning_rate = config.model.optimizer.initial_lr
        self.loss_type = config.model.loss_type
        self.metrics = config.model.metrics
        self.config = config

    def generate_training_model(self):

        """Generates the model for training, and returns the compiled model.

        :returns model: the final model which is ready to be trained.
        """

        raise Exception('not implemented')

    def get_inference_model(self):

        """Loads the best model and returns the ready-to-use model.





        :return:
        """


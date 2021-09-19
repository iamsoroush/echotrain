

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

        self.name = config.name
        self.optimizer_type = config.optimizer.type
        self.learning_rate = config.optimizer.initial_lr
        self.loss_type = config.loss_type
        self.metrics = config.metrics

    def generate_model(self):

        """Generates the model, and returns the compiled model.

        :returns model: the final model which is ready to be trained.
        """

        raise Exception('not implemented')

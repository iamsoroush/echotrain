

class BaseModel:

    def __init__(self, config):

        """

        :param config: dictionary of {config_name: config_value}
        """

        self.config = config
        self.name = config.name

    def generate_model(self):

        """Generates the model, and returns the compiled model.

        :returns model: the final model which is ready to be trained.
        """

        raise Exception('not implemented')

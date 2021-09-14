

class BaseModel:

    def __init__(self, name):
        self.name = name

    def generate_model(self):

        """Generates the model, and returns the compiled model.

        :returns model: the final model which is ready to be trained.
        """

        raise Exception('not implemented')

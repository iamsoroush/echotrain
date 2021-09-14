

class BaseModel:

    def __init__(self, name):
        self.name = name

    def generate_model(self):
        raise Exception('not implemented')

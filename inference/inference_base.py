

class InferenceBase:

    def __init__(self):
        pass

    def load_model(self):
        raise Exception('not implemented')

    def pre_process(self, image):
        raise Exception('not implemented')

    def post_process(self, image):
        raise Exception('not implemented')

    def process(self, image):
        raise Exception('not implemented')

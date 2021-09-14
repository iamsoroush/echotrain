

class TrainerBase:

    def __init__(self, checkpoints_dir, logs_dir):
        self.checkpoints_dir = checkpoints_dir
        self.logs_dir = logs_dir

    def train(self):
        pass

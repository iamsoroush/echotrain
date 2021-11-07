from abc import ABC,abstractmethod

class EFBase(ABC):

    def __init__(self, config):

        self.config = config
        self.estimation_method = config.estimation_method

    @abstractmethod
    def ef_estimation(self,ed_frame,es_frame):
        pass




from abc import ABC, abstractmethod


class EvaluatorBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def build_data_frame(self):

        """Generates a report as a pandas dataframe, each row represents a single data point (evluation image)."""

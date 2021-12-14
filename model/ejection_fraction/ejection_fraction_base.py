from abc import ABC,abstractmethod

class EFBase(ABC):

    def __init__(self, config):

        self.config = config
        self.estimation_method = config.estimation_method

    @abstractmethod
    def ef_estimation(self,ed_frame,es_frame):
        """
        This method estimate ejection fraction in percentage from end diastolic and end systolic frame.

        Args:
            ed_frame: end diastolic frame (in rp model should be label and in encoder model should be image)
            es_frame: end systolic frame (in rp model should be label and in encoder model should be image)

        Returns:
            ejection fraction in percentage
        """
        pass

    @abstractmethod
    def train(self):
        """
        This method train the model you want and can be handled from config.yaml file
        in config:
        model_type can be:

            'svr': SVR model
            'rfr': RandomForestRegressor model
            'knn': KNeighborsRegressor model
            'gbr': GradientBoostingRegressor model
            'dtr': DecisionTreeRegressor model
            'lr': LinearRegression model
            'mlp': MLPRegressor model
            'dr': DummyRegressor model
            'gpr': GaussianProcessRegressor model
            'sgdr': SGDRegressor model
            'abr': AdaBoostRegressor model

        Returns:trained model controlled in config.yaml file

        """
        pass

    @abstractmethod
    def export(self,model):
        """
        This method save the sklearn model trained by train method at the designated directory
        in config.(exported_dir:)

        Args:
            model: sklearn mode trained previously by train method
        """
        pass



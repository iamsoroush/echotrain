from abc import ABC, abstractmethod


class BaseClass(ABC):

    def __init__(self, config=None):
        self._set_defaults()
        if config is not None:
            self._load_params(config)

        self.config = config

    @abstractmethod
    def _load_params(self, config):

        """Load parameters using config file."""

        pass

    @abstractmethod
    def _set_defaults(self):

        """Default values for your class, if None is passed as config.

        Should initialize the same parameters as in ``_load_params`` method.
        """

        pass

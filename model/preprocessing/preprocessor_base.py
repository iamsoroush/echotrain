from abc import ABC, abstractmethod


class PreProcessor(ABC):
    """
     PreProcess module used for images, batches, generators

    Example::

        preprocessor = PreProcess()
        image = preprocessor.img_preprocess(image)
        X, y = preprocessor.batch_preprocess(batch_data)
        preprocessed_data_gen = preprocessor.add_preprocess(data_gen, add_augmentation=True)

    """

    def __init__(self, config=None):

        self._set_defaults()

        if config is not None:
            self._load_params(config)

    @abstractmethod
    def img_preprocess(self, image, inference=False):

        """
        pre-processing on input image

        :param image: input image, np.array
        :param inference: do inference-specific operations if the user is in inference phase

        :returns pre_processed_img: pre-processed image ready to be processed by your model.
        """

    @abstractmethod
    def batch_preprocess(self, batch):

        """Batch pre_processing.

        :param batch: input batch (X, y)

        :returns x_preprocessed_batch: preprocessed batch for x
        :returns y_preprocessed_batch: preprocessed batch for y
        """

    @abstractmethod
    def add_preprocess(self, generator, add_augmentation):

        """Provides the pre-processing functionality for given data generator

        :param generator: input generator ready for pre-processing, data generator < class DataGenerator >
        :param add_augmentation: pass True if your generator is train_gen

        :returns preprocessed_gen: preprocessed data generator.
        """

    @abstractmethod
    def _load_params(self, config):
        pass

    @abstractmethod
    def _set_defaults(self):
        pass

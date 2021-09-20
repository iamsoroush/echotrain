import tensorflow as tf
import numpy as np


class PreProcessor:
    """
     PreProcess module used for images, batches, generators

    HOW TO:
    preprocessor = PreProcess()
    image = preprocessor.img_preprocess(image)
    X, y = preprocessor.batch_preprocess(gen_batch)
    data_gen = preprocessor.add_preprocess(data_gen)
    """

    def __init__(self, target_size=(128, 128)):
        """
        :param target_size: image target size for resizing, tuple (image_height, image_width)
        """
        self.target_size = target_size

    def img_preprocess(self, image):
        """
        pre-processing on input image

        :param image: input image, np.array

        :return: pre_processed_img: pre-processed image
        """
        # 1. normalization on the given image
        normalized_image = self.rescaling(image, 1/255.)

        # preprocessed image
        pre_processed_img = normalized_image

        return pre_processed_img

    def batch_preprocess(self, batch):
        """
        batch pre_processing function

        :param batch: input batch (X, y)

        :return: x_preprocessed_batch: preprocessed batch for x
        :return: y_preprocessed_batch: preprocessed batch for y
        """
        # images of the give batch
        x = batch[0]

        # labels of the give batch
        y = batch[1]

        # pre-processing every image of the batch given
        x_preprocessed_batch = np.array(list(map(self.img_preprocess, x)))
        # the labels of the batches do not need pre-processing (yet!)
        y_preprocessed_batch = y

        return x_preprocessed_batch, y_preprocessed_batch

    def add_preprocess(self, generator):
        """providing the suggested pre-processing for the given generator

        :param generator: input generator ready for pre-processing, data generator < class DataGenerator >

        :return: preprocessed_gen: preprocessed generator, data generator < class DataGenerator >
        """
        # iterate through batches of the given generator
        for i, batch in enumerate(generator):
            # pre-processing and copying every batches
            generator.batch_gen_copy(self.batch_preprocess(batch), i)

        preprocessed_gen = generator
        return preprocessed_gen

    def resizing(self, image):
        """
        resizing image into the target_size dimensions

        :param image: input image, np.array

        :return: resized image
        """

        image_resized = np.array(image.resize(image,
                                              self.target_size,
                                              antialias=False,
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        return image_resized

    @staticmethod
    def rescaling(image, rescaling_ratio=1/255.):
        """
        rescaling the input image

        :param rescaling_ratio: rescaling ratio to apply on the input image, float
        :param image: input image, np.array

        :return: rescaled image
        """
        return image * rescaling_ratio

    def augmentation(self):

        raise Exception('not implemented')

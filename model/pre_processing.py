import tensorflow as tf
import numpy as np


class PreProcessor:
    """
    HOW TO:

    X, y = PreProcess().batch_pre_process(batch)
    image = PreProcess().pre_process(imge)
    """
    def __init__(self, target_size=(128, 128)):
        """

        :param target_size: image target size for resizing, tuple (image_height, image_width)
        """
        self.target_size = target_size

    def pre_process(self, image):
        """
        pre-processing on input image
        :param image: input image, np.array
        :return: pre_processed_img: pre-processed image
        """
        normalized_image = self.rescaling(image)

        pre_processed_img = normalized_image

        return pre_processed_img

    def batch_pre_process(self, batch):
        """
        batch pre_processing function
        :param batch: input batch (X, y)
        :return: x_preprocessed_batch: preprocessed batch for x
        :return: y_preprocessed_batch: preprocessed batch for y
        """
        x = batch[0]
        y = batch[1]

        x_preprocessed_batch = np.array(list(map(self.pre_process, x)))
        y_preprocessed_batch = y

        return x_preprocessed_batch, y_preprocessed_batch

    def resizing(self, image):
        """
        resizing image into the target_size dimensions

        :param image: input image, np.array
        :return: resized image
        """

        image_resized = tf.image.resize(image,
                                        self.target_size,
                                        antialias=False,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image_resized

    @staticmethod
    def rescaling(image):
        """
        rescaling the input image
        :param image: input image, np.array
        :return: rescaled image
        """
        return image/255.

    # def augmentation(self):

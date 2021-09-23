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
        self.normalization = True

    def img_preprocess(self, image):

        """
        pre-processing on input image

        :param image: input image, np.array

        :return: pre_processed_img: pre-processed image
        """

        # 1. normalization on the given image
        if self.normalization:
            pre_processed_img = self.rescaling(image, 1/255.)
        else:
            pre_processed_img = image

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

        while True:
            batch = next(generator)
            pre_processed_batch = self.batch_preprocess(batch)
            yield pre_processed_batch
        # pre_processed_gen = PreProcessedGen(generator, self.batch_preprocess)
        # return pre_processed_gen

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


class PreProcessedGen(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, generator, pre_processing):
        self.generator = generator
        self.pre_processing = pre_processing

    def __len__(self):
        return self.generator.get_n_iter()

    def __getitem__(self, idx):
        batch = self.generator[idx]
        preprocessed_batch = self.pre_processing(batch)
        return preprocessed_batch

    def on_epoch_end(self):
        self.generator.on_epoch_end()

    def __next__(self):

        """create a generator that iterate over the Sequence."""

        return self.next()

    def next(self):

        """
        Create iteration through batches of the generator
        :return: next batch, np.ndarray
        """

        index = next(self.generator.flow_index())
        return self.__getitem__(index)

    def reset(self):

        """reset indexes for iteration"""

        self.generator.reset()

    def random_visualization(self):

        """random visualization of an image"""

        # choosing random image from dataset random indexes
        random_batch_index = np.random.randint(self.__len__())
        random_batch = self.__getitem__(random_batch_index)
        random_image_index = np.random.randint(len(random_batch[0]))
        random_image = random_batch[0][random_image_index]
        image_label = random_batch[1][random_image_index]

        # setting a two-frame-image to plotting both the image and its segmentation labels
        self.generator.visualization(random_image, image_label)

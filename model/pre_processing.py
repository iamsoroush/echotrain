import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
import albumentations as A
import matplotlib.pyplot as plt


class PreProcessor:
    """
    PreProcess module used for images, batches, generators

    HOW TO:
    preprocessor = PreProcess()
    image = preprocessor.img_preprocess(image)
    X, y = preprocessor.batch_preprocess(gen_batch)
    data_gen = preprocessor.add_preprocess(data_gen, add_augmentation=True)
    """

    def __init__(self, config):

        """
        target_size: image target size for resizing, tuple (image_height, image_width)
        min: minimum value of the image range, int
        max: maximum value of the image range, int
        normalization: for rescaling the image, bool
        """
        self.config = config
        self.input_h = config.input_h
        self.input_w = config.input_w
        self.target_size = (self.input_h, self.input_w)
        self.max = config.pre_process.max
        self.min = config.pre_process.min
        self.do_resizing = config.pre_process.do_resizing
        self.do_normalization = config.pre_process.do_normalization

        # Augmentation
        self.augmentation_config = config.pre_process.augmentation
        # self.rotation_range = self.augmentation_config.rotation_range
        # self.flip_p = self.augmentation_config.flip_proba
        self.aug = Augmentation(self.augmentation_config)

    def img_preprocess(self, image):

        """
        pre-processing on input image

        :param image: input image, np.array

        :return: pre-processed-img
        """

        pre_processed_img = image.copy()

        # resizing
        if self.do_resizing:
            pre_processed_img = self._resize(pre_processed_img)

        # converting the images to grayscale
        if image.shape[-1] != 1 and len(image.shape) != 2:
            pre_processed_img = self._convert_to_gray(pre_processed_img)

        # normalization on the given image
        if self.do_normalization:
            pre_processed_img = self._rescaling(pre_processed_img, self.min, self.max)

        return pre_processed_img

    def label_preprocess(self, label):
        """
        pre-processing on input label

        :param label: input label, np.array

        :return: pre-processed label
        """
        # resizing
        if self.do_resizing:
            label = self._resize(label[:, :, tf.newaxis])

        return label

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

        # the labels of the batches do not need pre-processing (except for resizing if it is true)
        y_preprocessed_batch = np.array(list(map(self.label_preprocess, y)))

        return x_preprocessed_batch, y_preprocessed_batch

    def add_preprocess(self, generator, add_augmentation):

        """providing the suggested pre-processing for the given generator

        :param generator: input generator ready for pre-processing, data generator < class DataGenerator >
        :param add_augmentation: pass True if your generator is train_gen

        :return: preprocessed_gen: preprocessed generator, data generator < class DataGenerator >
        """

        while True:
            batch = next(generator)
            pre_processed_batch = self.batch_preprocess(batch)

            if add_augmentation:
                pre_processed_batch = self.aug.batch_augmentation(pre_processed_batch)

            yield pre_processed_batch
        # pre_processed_gen = PreProcessedGen(generator, self.batch_preprocess)
        # return pre_processed_gen

    def _resize(self, image):

        """
        resizing image into the target_size dimensions

        :param image: input image, np.array

        :return: resized image
        """

        image_resized = np.array(tf.image.resize(image,
                                                 self.target_size,
                                                 antialias=False,
                                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR))
        return image_resized

    @staticmethod
    def _rescaling(image, min_val, max_val):

        """
        rescaling the input image

        :param image: input image, np.array
        :param min_val: minimum value of the image
        :param max_val: maximum value of the image

        :return: rescaled image
        """

        rescaled_image = (image - min_val) / (max_val - min_val)

        return rescaled_image

    @staticmethod
    def _convert_to_gray(image):

        """
        converting the input image to grayscale, if needed

        :param image: input image, np array

        :return: converted image
        """

        gray_image = rgb2gray(image)
        return gray_image


class Augmentation:

    """
    This class is implementing augmentation on the batches of data

    How to:
    aug = Augmentation()
    data = aug.batch_augmentation(x,y)
    x = images(batch)
    y = masks(batch)
    data = augmented batch
    """

    def __init__(self, config):

        """
        :param config: augmentation part of config file: config.pre_process.augmentation, containing:
            rotation_range: the range limitation for rotation in augmentation
            flip_proba: probability for flipping
        """

        self.rotation_range = config.rotation_range
        self.flip_proba = config.flip_proba
        self.transform = A.Compose([
            A.Flip(p=self.flip_proba),
            A.ShiftScaleRotate(0, 0, border_mode=0, rotate_limit=self.rotation_range, p=0.5)
        ])

    def batch_augmentation(self, batch):

        """
        this function implement augmentation on batches
        :param x: batch images of the whole batch
        :param y: batch masks of the whole batch
        :return: x, y: the image and mask batches.
        """

        # changing the type of the images for albumentation
        x = batch[0]
        y = batch[1]

        x = x.astype('float32')

        # implementing augmentation on every image and mask of the batch
        for i in range(len(x)):
            transformed = self.transform(image=x[i], mask=y[i])
            x[i] = transformed['image']
            y[i] = transformed['mask']

        return x, y

    def add_augmentation(self, generator):
        """
        calling the batch_augmentation
        :param generator: the input of this class must be generator
        :yield: the batches of the augmented generator
        """

        while True:
            batch = next(generator)
            augmented_batch = self.batch_augmentation(batch)
            yield augmented_batch


class PreProcessedGen(tf.keras.utils.Sequence):
    """
    this class makes the class PreProcessor output, a generator and makes it suitable to fit to the model

    HOW TO:
    pre_processed_gen = PreProcessedGen(generator, self.batch_preprocess)

    """

    def __init__(self, generator, pre_processing):
        """
        :param generator: the output of the dataset_generator,generator <Class dataset_generator>
        :param pre_processing: preprocessing functions from the class PreProcessor
        """

        self.generator = generator
        self.pre_processing = pre_processing

    def __len__(self):
        """
        :return: number of the batches per epoch
        """

        return self.generator.get_n_iter()

    def __getitem__(self, idx):
        """
        :param idx: the index of the batch
        :return: preprocessed_batch
        """

        batch = self.generator[idx]
        preprocessed_batch = self.pre_processing(batch)
        return preprocessed_batch

    def on_epoch_end(self):
        """
        :return: shuffle at the end of each epoch
        """

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

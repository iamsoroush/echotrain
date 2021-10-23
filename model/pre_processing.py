import tensorflow as tf
import numpy as np
from skimage.color import rgb2gray
from .augmentation import Augmentation
import cv2
from skimage.restoration import denoise_wavelet, estimate_sigma
import scipy as sp
import scipy.ndimage as nd


class PreProcessor:
    """
     PreProcess module used for images, batches, generators

    Example::

        preprocessor = PreProcess()
        image = preprocessor.img_preprocess(image)
        X, y = preprocessor.batch_preprocess(gen_batch)
        data_gen = preprocessor.add_preprocess(data_gen, add_augmentation=True)

    Attributes:

        target_size: image target size for resizing, tuple (image_height, image_width)
        min: minimum value of the image range, int
        max: maximum value of the image range, int
        normalization: for rescaling the image, bool

    """

    def __init__(self, config=None):

        """
        """

        self._load_params(config)

        # GaussianBlur
        self.do_GaussianBlur = False
        # MedianBlur
        self.do_medianBlur = True
        # Wavelet Transformation Denoising
        self.do_waveletDenoising = True
        # Histogram Equalization
        self.do_histEqualizing = False
        # Edge Detection
        self.do_edgeDetection = False
        # Edge Enhancement
        self.do_EdgeEnhancement = True
        # Contras Enhancement
        self.do_contrastEnhancement = False
        # Morphology
        self.do_morphology = True
        # LoG Filtering
        self.do_LoG = False
        # thresholding
        self.do_thresholding = False
        # Clipping
        self.do_clipping = False
        # Gamma Correction
        self.do_gammaCorrection = False
        # Augmentation
        self.aug = Augmentation(config)

    def img_preprocess(self, image, inference=False):

        """
        pre-processing on input image

        :param image: input image, np.array
        :param inference: resize if the user is in inference phase

        :return: pre_processed_img
        """

        pre_processed_img = image.copy()

        # converting the images to grayscale
        if len(image.shape) != 2 and image.shape[-1] != 1:
            pre_processed_img = self._convert_to_gray(pre_processed_img)

        # resizing
        if self.do_resizing or inference:
            pre_processed_img = self._resize(pre_processed_img)

        # GaussianBlur
        if self.do_GaussianBlur:
            pre_processed_img = cv2.GaussianBlur(pre_processed_img, (5, 5), 3)

        # MedianBlur
        if self.do_medianBlur:
            pre_processed_img = cv2.medianBlur(pre_processed_img, 5)

        # Wavelet Transformation Denoising
        if self.do_waveletDenoising:
            est_sigma = estimate_sigma(pre_processed_img, average_sigmas=True)
            pre_processed_img = denoise_wavelet(pre_processed_img, wavelet_levels=None,
                                                multichannel=False, convert2ycbcr=False,
                                                method='VisuShrink', mode='soft',
                                                sigma=est_sigma/4, rescale_sigma=True)

        # histogram equalization
        if self.do_histEqualizing:
            pre_processed_img = cv2.equalizeHist(pre_processed_img)

        # edge sharpening kernel
        if self.do_EdgeEnhancement:
            kernel = np.array([[-0.1, -0.2, -0.1],
                               [-0.2, 1.2, -0.2],
                               [-0.1, -0.2, -0.1]])
            pre_processed_img = pre_processed_img[:, :]+np.abs(cv2.filter2D(src=pre_processed_img, ddepth=-1, kernel=kernel))
            # print(pre_processed_img.shape)
            # for i in range(pre_processed_img.shape[0]):
            #     for j in range(pre_processed_img.shape[1]):
            #         pre_processed_img[i][j] += np.abs(edges[i][j])
            # print(pre_processed_img.shape)

        # opening and closing morphology
        if self.do_morphology:
            pre_processed_img = cv2.morphologyEx(pre_processed_img, cv2.MORPH_OPEN, (1, 1))
            pre_processed_img = cv2.morphologyEx(pre_processed_img, cv2.MORPH_CLOSE, (1, 1))
            pre_processed_img = cv2.morphologyEx(pre_processed_img, cv2.MORPH_CLOSE, (3, 3))

        # edge detection
        if self.do_edgeDetection:
            pre_processed_img = self._canny_detector(pre_processed_img)

        # LoG Filtering
        if self.do_LoG:
            pre_processed_img = self.LoG_filter(pre_processed_img)

        # contrast enhancement
        if self.do_contrastEnhancement:
            clahe = cv2.createCLAHE(clipLimit=40)
            pre_processed_img = clahe.apply(pre_processed_img)

        if self.do_thresholding:
            ret, pre_processed_img = cv2.threshold(pre_processed_img, 0, 255, cv2.THRESH_TRUNC)

        # clipping
        if self.do_clipping:
            pre_processed_img = np.clip(pre_processed_img,
                                        pre_processed_img.mean(),
                                        pre_processed_img.max())

        # gamma correction
        if self.do_gammaCorrection:
            gamma = 0.5
            pre_processed_img = np.clip(pow((pre_processed_img / 255.0), gamma) * 255.0, 0, 255)

        # normalization on the given image
        if self.do_normalization:
            pre_processed_img = self._rescale(pre_processed_img,
                                              pre_processed_img.min(),
                                              pre_processed_img.max())

        return pre_processed_img.astype('float32')

    def label_preprocess(self, label):

        """
        pre-processing on input label

        :param label: input label, np.array

        :return: pre-processed label
        """

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
        # the labels of the batches do not need pre-processing (yet!)
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

    def _load_params(self, config):
        self._set_defaults()

        if config is not None:
            self.input_h = config.input_h
            self.input_w = config.input_w
            self.max = config.pre_process.max
            self.min = config.pre_process.min
            self.do_resizing = config.pre_process.do_resizing
            self.do_normalization = config.pre_process.do_normalization

    def _set_defaults(self):
        self.input_h = 256
        self.input_w = 256
        self.max = 255
        self.min = 0
        self.do_resizing = True
        self.do_normalization = True

    @property
    def target_size(self):
        return self.input_h, self.input_w

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
    def _rescale(image, min_val, max_val):

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

    def _contrast_enhancement(self, image):
        raise Exception('not implemented')

    @staticmethod
    def _canny_detector(img, weak_th=None, strong_th=None):

        # conversion of image to grayscale
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Noise reduction step
        # img = cv2.GaussianBlur(img, (5, 5), 1.4)

        # Calculating the gradients
        gx = cv2.Sobel(np.float32(img), cv2.CV_64F, 1, 0, 3)
        gy = cv2.Sobel(np.float32(img), cv2.CV_64F, 0, 1, 3)

        # Conversion of Cartesian coordinates to polar
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)

        # setting the minimum and maximum thresholds
        # for double thresholding
        mag_max = np.max(mag)
        if not weak_th: weak_th = mag_max * 0.3
        if not strong_th: strong_th = mag_max * 0.6

        # getting the dimensions of the input image
        height, width = img.shape

        # Looping through every pixel of the grayscale
        # image
        for i_x in range(width):
            for i_y in range(height):

                grad_ang = ang[i_y, i_x]
                grad_ang = abs(grad_ang - 180) if abs(grad_ang) > 180 else abs(grad_ang)

                # selecting the neighbours of the target pixel
                # according to the gradient direction
                # In the x axis direction
                if grad_ang <= 22.5:
                    neighb_1_x, neighb_1_y = i_x - 1, i_y
                    neighb_2_x, neighb_2_y = i_x + 1, i_y

                # top right (diagonal-1) direction
                elif grad_ang > 22.5 and grad_ang <= (22.5 + 45):
                    neighb_1_x, neighb_1_y = i_x - 1, i_y - 1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y + 1

                # In y-axis direction
                elif grad_ang > (22.5 + 45) and grad_ang <= (22.5 + 90):
                    neighb_1_x, neighb_1_y = i_x, i_y - 1
                    neighb_2_x, neighb_2_y = i_x, i_y + 1

                # top left (diagonal-2) direction
                elif grad_ang > (22.5 + 90) and grad_ang <= (22.5 + 135):
                    neighb_1_x, neighb_1_y = i_x - 1, i_y + 1
                    neighb_2_x, neighb_2_y = i_x + 1, i_y - 1

                # Now it restarts the cycle
                elif grad_ang > (22.5 + 135) and grad_ang <= (22.5 + 180):
                    neighb_1_x, neighb_1_y = i_x - 1, i_y
                    neighb_2_x, neighb_2_y = i_x + 1, i_y

                # Non-maximum suppression step
                if width > neighb_1_x >= 0 and height > neighb_1_y >= 0:
                    if mag[i_y, i_x] < mag[neighb_1_y, neighb_1_x]:
                        mag[i_y, i_x] = 0
                        continue

                if width > neighb_2_x >= 0 and height > neighb_2_y >= 0:
                    if mag[i_y, i_x] < mag[neighb_2_y, neighb_2_x]:
                        mag[i_y, i_x] = 0

        weak_ids = np.zeros_like(img)
        strong_ids = np.zeros_like(img)
        ids = np.zeros_like(img)

        # double thresholding step
        for i_x in range(width):
            for i_y in range(height):

                grad_mag = mag[i_y, i_x]

                if grad_mag < weak_th:
                    mag[i_y, i_x] = 0
                elif strong_th > grad_mag >= weak_th:
                    ids[i_y, i_x] = 1
                else:
                    ids[i_y, i_x] = 2

        # finally returning the magnitude of
        # gradients of edges

        final_img = np.where(mag == 0, 0.7, 1) * img
        return final_img

    def LoG_filter(self, image):
        LoG = nd.gaussian_laplace(image, 2)
        thres = np.absolute(LoG).mean() * 0.25
        output = sp.zeros(LoG.shape)
        w = output.shape[1]
        h = output.shape[0]

        for y in range(1, h - 1):
            for x in range(1, w - 1):
                patch = LoG[y - 1:y + 2, x - 1:x + 2]
                p = LoG[y, x]
                maxP = patch.max()
                minP = patch.min()
                if (p > 0):
                    zeroCross = True if minP < 0 else False
                else:
                    zeroCross = True if maxP > 0 else False
                if ((maxP - minP) > thres) and zeroCross:
                    output[y, x] = 1

        final_img = np.where(output == 0, 0.2, 1) * image
        return final_img

from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Concatenate, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

from .model_base import ModelBase
from .metric import get_iou_coef, get_dice_coeff
from .loss import dice_coef_loss


class UNet(ModelBase):

    def __init__(self, config):

        """
        HOWTO:
        first you have to call UNet class: unet=UNet(config=config)
        then 2 methods have been provided:
                get_model_graph(): this method gives you non_compiled unet model
                generate_training_model(): this method gives you compiled model with
                attributes designated in config.(you can see options you have at Attributes section below.)


        :param config:

        Attributes:
            input_h:height of your image
            input_w:width of your image
            n_channels:number of channels of image
            optimizer:
                type: can be "adam"
                initial_lr:default would be 0.001
            metrics: 'acc', 'dice_coef', 'iou', '2d_hausdorff' is supported
            loss:'binary_crossentropy', 'dice_coef_loss' is supported
        """

        super(UNet, self).__init__(config=config)

        try:
            self.optimizer_type = config.model.optimizer.type
        except AttributeError:
            self.optimizer_type = 'adam'

        try:
            self.learning_rate = config.model.optimizer.initial_lr
        except AttributeError:
            self.learning_rate = 0.001

        try:
            self.loss_type = config.model.loss_type
        except AttributeError:
            self.loss_type = 'binary_crossentropy'

        try:
            self.metrics = config.model.metrics
        except AttributeError:
            self.metrics = ['iou']

    def generate_training_model(self):
        """
        compile model from get_model_graph method with the optimizer,
        metrics and loss function written in config_example.yaml file

        :return: compiled model
        """
        model = self.get_model_graph()
        optimizer = self._get_optimizer()
        metrics = self._get_metrics()
        loss = self._get_loss()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def get_model_graph(self):
        """
        create a unet model for images with input_h height, input_w width
        and n_channels channels written in config_example.yaml file
        :return:unet model with 24 conv layer
        """
        inputs = Input((self.input_h, self.input_w, self.n_channels))
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv5))
        merge6 = Concatenate(axis=3)([conv4, up6])
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

        up7 = Conv2D(256, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv6))
        merge7 = Concatenate(axis=3)([conv3, up7])
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

        up8 = Conv2D(128, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv7))
        merge8 = Concatenate(axis=3)([conv2, up8])
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(conv8))
        merge9 = Concatenate(axis=3)([conv1, up9])
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

        model = Model(inputs=inputs, outputs=conv10)

        return model

    def _get_metrics(self):
        """
        :return:all metrics chosen in config file in a python list suitable for compile method of keras
        """
        metrics = []
        if 'iou' in self.metrics:
            metrics.append(get_iou_coef(threshold=self.inference_threshold))
        if 'acc' in self.metrics:
            metrics.append('acc')
        if 'dice_coef' in self.metrics:
            metrics.append(get_dice_coeff(threshold=self.inference_threshold))
        # if '2d_hausdorff' in self.metrics:
        #     metrics.append(directed_hausdorff)
        return metrics

    def _get_optimizer(self):
        """

        :return:the optimizer with learning rate that were designated in config file
        """
        if self.optimizer_type == 'adam':
            return Adam(learning_rate=self.learning_rate)

    def _get_loss(self):
        """

        :return:the loss function that is designated in config file
        """
        if self.loss_type == 'binary_crossentropy':
            return 'binary_crossentropy'
        if self.loss_type == 'dice_coef_loss':
            return self._dice_coef_loss

    def _iou_coef(self, y_true, y_pred, smooth=1):
        """

         :param y_true: label image from the dataset
         :param y_pred: model segmented image prediction
         :param smooth:
         :return:calculate Intersection over Union for y_true and y_pred
         """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3]) - intersection
        iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
        return iou

    def _dice_coef(self, y_true, y_pred, smooth=1):
        """

         :param y_true: label image from the dataset
         :param y_pred: model segmented image prediction
         :param smooth:
         :return: calculate dice coefficient between y_true and y_pred
         """
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

    def _dice_coef_loss(self, y_true, y_pred):
        """

         :param y_true: label image from the dataset
         :param y_pred: model segmented image prediction
         :return: dice coefficient loss function
         """
        return -1 * (self._dice_coef(y_true, y_pred))

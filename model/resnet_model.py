from .model_base import ModelBase
import segmentation_models as sm
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Concatenate, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from .metric import get_iou_coef, get_dice_coeff, get_soft_dice, get_hausdorff_distance,get_mad,get_soft_iou
from .loss import dice_coef_loss,iou_coef_loss,soft_iou_loss,soft_dice_loss

sm.set_framework('tf.keras')
sm.framework()

class ResNetUNet(ModelBase):
    def __init__(self,config):

        super(ResNetUNet, self).__init__(config=config)
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

    def get_model_graph(self,isfreez=True):

        resnet_unet=sm.Unet('resnet34',encoder_weights='imagenet',input_shape=(256 ,256 ,3),encoder_freeze=isfreez)
        input=Input((256,256,3))
        output=resnet_unet(input)
        model=Model(input,output)

        return model

    def _get_metrics(self):
        metrics = []
        if 'iou' in self.metrics:
            metrics.append(get_iou_coef())
        if 'dice_coef' in self.metrics:
            metrics.append(get_dice_coeff())
        if 'acc' in self.metrics:
            metrics.append('acc')
        if 'soft_iou' in self.metrics:
            metrics.append(get_soft_iou())
        if 'soft_dice' in self.metrics:
            metrics.append(get_soft_dice())
        if 'hausdorf' in self.metrics:
            metrics.append(get_hausdorff_distance())
        if 'mad' in self.metrics:
            metrics.append(get_mad())

        return metrics

    def _get_loss(self):

        if self.loss_type == 'binary_crossentropy':
            return 'binary_crossentropy'
        if self.loss_type=='dice_coef_loss':
            return dice_coef_loss
        if self.loss_type=='iou_coef_loss':
            return iou_coef_loss
        if self.loss_type=='soft_iou_loss':
            return soft_iou_loss
        if self.loss_type=='soft_dice_loss':
            return soft_dice_loss

    def _get_optimizer(self):

        if self.optimizer_type == 'adam':
            return Adam(learning_rate=self.learning_rate)
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

from .model_base import ModelBase
from .metric import get_dice_coeff, get_iou_coef
from .loss import dice_coef_loss


class UNetBaseline(ModelBase):
    """Unet V2 implementation from CAMUS paper.

    from the paper:

        - model input: (256, 256, 1)
        - density normalization after resizing
        - no augmentation
        - weight init: glorot_uniform
        - max_pool: (2, 2)
        - lowest resolution: (16, 16)
        - activation: relu
        - final activation: softmax (we used sigmoid because of binary output)
        - optimizer: Adam(lr=1e-4)
        - loss: crossentropy and weight decay(L2 regularization of the weights). we will not apply weight decay, we will
          try regularization if the model over-fits
        - epochs: 30
        - params: ~18M
        - batch size: 10

    """

    def __init__(self, config, hp):
        super().__init__(config)
        self.config = config
        self.hp = hp
        self._read_config()
        #conv_kernel_size = hp.Int('conv_kernel_size', min_value=2, max_value=7, step=1)
        self.conv_kernel_size = (3, 3)
        self.conv_padding = 'same'

        #conv_trans_kernel_size = hp.Int('conv_trans_kernel_size', min_value=2, max_value=7, step=1)
        self.conv_trans_kernel_size = (3, 3)
        self.conv_trans_strides = (2, 2)
        self.conv_trans_padding = 'same'

        self.max_pool_size = (2, 2)
        self.max_pool_strides = (2, 2)

        activation_function = hp.Choice("activation_function", ['relu', 'elu', 'tanh'],
                                        default=config.model.activation_function)
        self.activation = activation_function
        self.final_activation = 'sigmoid'

        kernel_initializer_selection = hp.Choice('kernel_initializer',
                                            ['random_normal', 'random_uniform', 'glorot_normal', 'glorot_uniform',None],
                                                 default=config.model.kernel_initializer)
        if kernel_initializer_selection == 'random_normal':
            self.kernel_initializer = tfk.initializers.RandomNormal()
        if kernel_initializer_selection == 'random_uniform':
            self.kernel_initializer = tfk.initializers.RandomUniform()
        if kernel_initializer_selection == 'glorot_normal':
            self.kernel_initializer = tfk.initializers.GlorotNormal()
        if kernel_initializer_selection == 'glorot_uniform':
            self.kernel_initializer = tfk.initializers.GlorotUniform()
        if kernel_initializer_selection is None:
            self.kernel_initializer = None

        kernel_regularizer_selection = hp.Choice('kernel_regularizer',
                                            ['l1', 'l2', 'l1_l2',None],
                                                 default=config.model.keras_regularizer)
        if kernel_regularizer_selection == 'random_normal':
            self.kernel_regularizer = tfk.regularizers.L1 ()
        if kernel_regularizer_selection == 'random_uniform':
            self.kernel_regularizer = tfk.regularizers.l2()
        if kernel_regularizer_selection == 'glorot_normal':
            self.kernel_regularizer = tfk.regularizers.l1_l2()
        if kernel_regularizer_selection == None:
            self.kernel_regularizer = None


    def generate_training_model(self):

        """
        compile model from get_model_graph method with the optimizer,
        metrics and loss function written in config_example.yaml file

        :return model: compiled model
        """

        model = self.get_model_graph()
        optimizer = self._get_optimizer()
        metrics = self._get_metrics()
        loss = self._get_loss()
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    def get_model_graph(self):

        """Defines the model's graph.

        Output of this method will be used for inference (by loading saved checkpoints) and training (by compiling the
        graph)

        :return model: a model of type tensorflow.keras.Model
            input_shape:(input_h, input_w, 1)
            output_shape:(input_h, input_w, 1)
        """
        input_tensor = tfk.Input((self.input_h, self.input_w, self.n_channels), name='input_tensor')

        # Encoder
        connection1, x = self._encoder_block(48, 1)(input_tensor)
        connection2, x = self._encoder_block(96, 2)(x)
        connection3, x = self._encoder_block(192, 3)(x)
        connection4, x = self._encoder_block(384, 4)(x)

        # Middle
        x = self._conv2d_bn_relu(768, 'middle_block1')(x)
        x = self._conv2d_bn_relu(768, 'middle_block2')(x)

        # Decoder
        x = self._decoder_transpose_block(384, 384, 1, connection=connection4)(x)
        x = self._decoder_transpose_block(192, 192, 2, connection=connection3)(x)
        x = self._decoder_transpose_block(96, 96, 3, connection=connection2)(x)
        x = self._decoder_transpose_block(48, 48, 4, connection=connection1)(x)

        # Output
        n_classes = 1
        x = tfkl.Conv2D(filters=n_classes,
                        kernel_size=(1, 1),
                        padding='same',
                        use_bias=True,
                        kernel_initializer=self.kernel_initializer,
                        name='final_conv')(x)
        x = tfkl.Activation(self.final_activation, name='output_tensor')(x)

        model = tfk.Model(input_tensor, x)
        return model

    def _read_config(self):

        """Tries to read parameters from config"""

        try:
            self.optimizer_type = self.hp.Choice('optimizer_type',['adam','sgd','rmsprop','adagrad'],
                                                 default=self.config.model.optimizer.type)
        except AttributeError:
            self.optimizer_type = 'adam'

        try:
            self.learning_rate = self.hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log",
                                               default=self.config.model.optimizer.initial_lr)
        except AttributeError:
            self.learning_rate = 0.001

        try:
            self.loss_type = self.hp.Choice('loss_function', ['binary_crossentropy', 'dice_coef_loss'],
                                            default=self.config.model.loss_type)
        except AttributeError:
            self.loss_type = 'binary_crossentropy'

        try:
            self.metrics = self.config.model.metrics
        except AttributeError:
            self.metrics = ['iou']

    def _get_metrics(self):

        """
        :return:all metrics chosen in config file in a python list suitable for compile method of keras
        """

        metrics = []
        if 'iou' in self.metrics:
            metrics.append(get_iou_coef(threshold=self.inference_threshold))
        if 'dice_coef' in self.metrics:
            metrics.append(get_dice_coeff(threshold=self.inference_threshold))
        # if '2d_hausdorff' in self.metrics:
        #     metrics.append(hausdorff)
        # if 'mean_absolute_distance' in self.metrics:
        #     metrics.append(mean_absolute_distance)
        return metrics

    def _get_optimizer(self):

        """

        :return:the optimizer with learning rate that were designated in config file
        """

        if self.optimizer_type == 'adam':
            return tfk.optimizers.Adam(learning_rate=self.learning_rate)
        if self.optimizer_type == 'sgd':
            return tfk.optimizers.SGD(learning_rate=self.learning_rate)
        if self.optimizer_type == 'rmsprop':
            return tfk.optimizers.RMSprop(learning_rate=self.learning_rate)
        if self.optimizer_type == 'adagrad':
            return tfk.optimizers.Adagrad(learning_rate=self.learning_rate)

    def _get_loss(self):
        """

        :return:the loss function that is designated in config file
        """
        if self.loss_type == 'binary_crossentropy':
            return 'binary_crossentropy'
        if self.loss_type == 'dice_coef_loss':
            return dice_coef_loss

    def _conv2d_bn_relu(self, filters, block_name):

        """Extension of Conv2D layer with BatchNormalization and ReLU activation"""

        conv_name = block_name + '_conv'
        act_name = block_name + '_' + self.activation
        bn_name = block_name + '_bn'

        def wrapper(input_tensor):
            x = tfkl.Conv2D(
                filters=filters,
                kernel_size=self.conv_kernel_size,
                padding=self.conv_padding,
                activation=None,
                use_bias=False,
                kernel_initializer=self.kernel_initializer,
                name=conv_name,
                kernel_regularizer=
            )(input_tensor)

            x = tfkl.BatchNormalization(name=bn_name)(x)
            x = tfkl.Activation(self.activation, name=act_name)(x)

            return x

        return wrapper

    def _convtranspose_bn_relu(self, filters, block_name):

        """Extension of Conv2DTranspose layer with BatchNormalization and ReLU activation"""

        transconv_name = block_name + '_conv2dtrans'
        act_name = block_name + '_' + self.activation
        bn_name = block_name + '_bn'

        def wrapper(input_tensor):
            x = tfkl.Conv2DTranspose(
                filters,
                kernel_size=self.conv_trans_kernel_size,
                strides=self.conv_trans_strides,
                padding=self.conv_trans_padding,
                activation=None,
                name=transconv_name,
                use_bias=False,
                kernel_initializer=self.kernel_initializer)(input_tensor)

            x = tfkl.BatchNormalization(name=bn_name)(x)
            x = tfkl.Activation(self.activation, name=act_name)(x)

            return x

        return wrapper

    def _decoder_transpose_block(self, transpose_filters, conv_filters, block, connection=None):

        """Generates a decoder block.

        ---------- convtranse_bn_relu + concat + conv2d_bn_relu + conv2d_bn_relu -----------
        """

        transpose_name = f'decoder_block{block}a'

        conv1_name = f'decoder_block{block}b'
        conv2_name = f'decoder_block{block}c'

        concat_name = f'decoder_block{block}_concat'

        def wrapper(input_tensor):
            x = self._convtranspose_bn_relu(transpose_filters, transpose_name)(input_tensor)
            if connection is not None:
                x = tfkl.Concatenate(axis=3, name=concat_name)([x, connection])

            x = self._conv2d_bn_relu(conv_filters, conv1_name)(x)
            x = self._conv2d_bn_relu(conv_filters, conv2_name)(x)

            return x

        return wrapper

    def _encoder_block(self, filters, block):

        """Generates an encoder block.

        ---------- conv2d_bn_relu + conv2d_bn_relu + max-pooling2d -----------
        """

        conv1_name = f'encoder_block{block}a'
        conv2_name = f'encoder_block{block}b'
        max_pool_name = f'encoder_block{block}_mp'

        def wrapper(input_tensor):
            x = self._conv2d_bn_relu(filters, conv1_name)(input_tensor)
            connection = self._conv2d_bn_relu(filters, conv2_name)(x)
            x = tfkl.MaxPool2D(self.max_pool_size, strides=self.max_pool_strides, name=max_pool_name)(connection)

            return connection, x

        return wrapper

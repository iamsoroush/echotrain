from base_model import BaseModel
from utils.handling_yaml import load_config_file
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D,Concatenate,Input,MaxPooling2D,UpSampling2D


class UNet(BaseModel):

    def init(self, config):

        """



        :param config:

        Attributes:
            input_h:description of input_h
        """

        super(UNet, self).init(config=config)
        '''try:
            self.optimizer_type = (config...)

        except AttributeError as e:
            self.optimizer_type = 'adam'

        try:
            self.loss_type

        self.input_h = config.input_h'''

   #def _load_attributes(self):
   #     pass

    def generate_training_model(self):
        model = self.get_model_graph()

        model.compile(optimizer='adam',loss=loss,metrics=['acc'])

        #metrics = self.get_metrics()
        #loss = self.get_loss()



    def get_model_graph(self):
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
        pass
path="../config/config_example.yaml"
config=load_config_file(path)
print(UNet(config=config).generate_training_model())
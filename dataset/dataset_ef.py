from pydoc import locate
import numpy as np
from dataset.dataset_generator import DatasetGenerator
from model.pre_processing import PreProcessor


class EfDataset:
    """
    create dataset for ef

    """

    def __init__(self, config):
        self.config = config
        self._load_config(config)

    def _load_config(self, config):

        """Load all parameters from config file"""

        if config is not None:
            cfg_dh = config.data_handler
            self.stage = cfg_dh.dataset_features.stage
            self.view = cfg_dh.dataset_features.view
            self.batch_size = cfg_dh.batch_size
            self.input_h = config.input_h
            self.input_w = config.input_w
            self.n_channels = config.n_channels
            self.dataset_name = config.data_handler.target_dataset_name
            self.dataset_class_path = config.dataset_class_path

    def load_dataframe(self, subset):

        dataset_class = locate(self.dataset_class_path)
        dataset = dataset_class(self.config)

        if subset == 'train':
            dataframe = dataset.train_df
        elif subset == 'test':
            dataframe = dataset.test_df_
        else:
            dataframe = dataset.val_df_

        return dataframe

    def create_x_y(self, type, subset):

        """

        sebset should be one of trian , test , val

        """

        dataframe = self.load_dataframe(subset)

        x = []
        y = []

        if type == 'image':

            for i in dataframe.iloc[:].iterrows():
                image_path = i[1].image_path
                x.append(image_path)
                if i[1].stage == 'ED':
                    label = i[1].lv_edv
                else:
                    label = i[1].lv_esv
                y.append(label)
        else:
            for i in dataframe.iloc[:].iterrows():
                label_path = i[1].label_path
                x.append(label_path)
                if i[1].stage == 'ED':
                    label = i[1].lv_edv
                else:
                    label = i[1].lv_esv
                y.append(label)
        return x, y

    def prepare_x_y(self, x, y, type):
        gen = DatasetGenerator(x, y, self.batch_size, (self.input_h, self.input_w), self.n_channels)

        preprocessor = PreProcessor(self.config)

        x = gen.generate_x(x)
        if type == 'image':
            x = np.array(list(map(preprocessor.img_preprocess, x)))
        else:
            x = np.array(list(map(preprocessor.label_preprocess, x)))

        y = np.expand_dims(y, -1)

        return x, y

    def ef_dataset(self, type, subset):

        x, y = self.create_x_y(type, subset)
        x, y = self.prepare_x_y(x, y, type)
        return x, y

from pydoc import locate
import numpy as np
from dataset.dataset_generator import DatasetGenerator
from model.pre_processing import PreProcessor


class EfDataset:
    """
    This class make dataset for EF models
    Need a config file as input
    The dataset can be based on image frames of ES and ED like :
            x : images , y : volumes (y is volume of ES OR EV based on image state)
    or dataset can be based on label(mask) frames of ES and ED like :
            x : label(mask) , y : volumes (y is volume of ES OR EV based on image state)
    x and y types are numpy.ndarray
    and dataset can be created for subsets   {train . test , val }
    Example :
            EfDataset=EfDataset(config_file)
            x_train , y_train = EfDataset.ef_dataset('image' , 'train')
            x_val , y_val = EfDataset.ef_dataset('image' , 'val')
            or
            x_train , y_train = EfDataset.ef_dataset('label' , 'train')
            x_val , y_val = EfDataset.ef_dataset('label' , 'val')

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
            self.dataset_class_path = config.dataset_class

    def load_dataframe(self, subset):
        """
        this method create dataframe based on subset and database
        database type (camus or echonet) will be read from config file
        input :
        subset(string) : 'train' or 'test' or 'val'
        return :
        dataframe


        """

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
        this method create a list of image or label(mask) path and a list of EC , ED volumes
        it uses dataframes for collecting path and volumes
        input :
        type(string) : 'image' or 'label'
        subset(string) : 'train' or 'test' or 'val'
        return :
        list x(list if type = image , dic if type = label) , y(list) , main_y (dic)

        """
        dataframe = self.load_dataframe(subset)

        y = []
        if type == 'image':
            x = []
            main_y = {}
            for i in dataframe.iloc[:].iterrows():
                image_path = i[1].image_path
                label_path = i[1].label_path
                x.append(image_path)
                main_y[image_path] = label_path

                if i[1].stage == 'ED':
                    label = i[1].lv_edv
                else:
                    label = i[1].lv_esv
                y.append(label)
        else:
            x = {}
            main_y = {}
            for i in dataframe.iloc[:].iterrows():
                image_path = i[1].image_path
                label_path = i[1].label_path
                x[image_path] = label_path
                main_y[image_path] = label_path

                if i[1].stage == 'ED':
                    label = i[1].lv_edv
                else:
                    label = i[1].lv_esv
                y.append(label)
        return x, y, main_y

    def prepare_x_y(self, x, y, main_y, type):
        """
        This method load images or labels(masks) from directory path
        we use DatasetGenerator's generate_x for image type and generate_y for label type for loading data
        we use PreProcessor class
        at the end we convert x , y from list to numpy array
        input :
        x (list if type = image , dic if type = label)  of images or labels(mask) path
        y (list) list of volumes
        y_main(dic) a dictionary of images and masks useful for generator class
        type(string) : 'image' , 'label'
        return :
        x (np.ndarray) , y(np.ndarray)

        """
        gen = DatasetGenerator(x, main_y, self.batch_size, (self.input_h, self.input_w), self.n_channels)

        preprocessor = PreProcessor(self.config)

        if type == 'image':
            x = gen.generate_x(x)
            x = np.array(list(map(preprocessor.img_preprocess, x)))
        else:
            x = gen.generate_y(x)
            x = np.array(list(map(preprocessor.label_preprocess, x)))

        y = np.expand_dims(y, -1)

        return x, y

    def ef_dataset(self, type, subset):
        """
        this method created dataset
        input :
        type(string) : 'image' or 'label'
        subset(string) : 'train' or 'test' or 'val'
        return :
        x (np.ndarray) , y(np.ndarray)

        example :
        x.shape : (2552, 256, 256, 1)
        y.shape : (2552, 1)


        """

        x, y, main_y = self.create_x_y(type, subset)
        x, y = self.prepare_x_y(x, y, main_y, type)
        return x, y

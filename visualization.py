import os
import matplotlib.pyplot as plt
import pandas as pd
import skimage.io as io


class Visualisation:

    def __init__(self):
        pass

    @staticmethod
    def load_bad_images(dataframe, column, value):
        """

        Args:
            dataframe: a pandas dataframe
            column: (string)
            value: it depends on dataframe value of column type , it can be float and string

        Returns:
            the specific image and mask will be ploted

        """
        x = dataframe[dataframe[column] == value]
        if x.empty:
            if type(value) is float:
                for i in dataframe[column].values:
                    if abs(value - i) < 0.000001:
                        value = i
        x = dataframe[dataframe[column] == value]
        if x.empty:
            raise Exception('null result , column or value could not be found')

        for i in x.iloc[:].iterrows():
            image_path = i[1].image_path
            label_path = i[1].label_path
            img = io.imread(os.path.join(image_path), plugin='simpleitk').astype('uint8')
            label = io.imread(os.path.join(label_path), plugin='simpleitk').astype('uint8')

            fig, ax = plt.subplots(1, 2)
            fig.set_size_inches(10, 5)
            ax[0].imshow(img, cmap='gray')
            ax[0].axis('off')
            ax[1].imshow(label, cmap='gray')
            ax[1].axis('off')
            print(i[1].case_id)
            plt.title(i[1].case_id + '_' + i[1].stage)
            plt.show()

    @staticmethod
    def load_bad_images_by_index(dataframe, indexes):
        """

        Args:
            dataframe: a pandas dataframe
            indexes: (list) list of dataframe's indexes

        Returns:
            the specific image and mask will be ploted

        """
        if type(indexes) is not list:
            raise Exception('indexes input type is not correct')

        for index in indexes:

            x = dataframe[dataframe.index == index]
            if x.empty:
                print(' null result , index {}  could not be found '.format(index))

            for i in x.iloc[:].iterrows():
                image_path = i[1].image_path
                label_path = i[1].label_path
                img = io.imread(os.path.join(image_path), plugin='simpleitk').astype('uint8')
                label = io.imread(os.path.join(label_path), plugin='simpleitk').astype('uint8')

                fig, ax = plt.subplots(1, 2)
                fig.set_size_inches(10, 5)
                ax[0].imshow(img, cmap='gray')
                ax[0].axis('off')
                ax[1].imshow(label, cmap='gray')
                ax[1].axis('off')
                print(i[1].case_id)
                plt.title(i[1].case_id + '_' + i[1].stage)
                plt.show()

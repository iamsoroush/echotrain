from .dataset_base import DatasetBase

MAPPER = {'dataset_base': DatasetBase}


def get_dataset_by_name(name):

    """
    Returns the dataset class related to the given name
    :param name: name of dataset class
    :returns dataset: a dataset class of type DatasetBase
    """

    return MAPPER[name]


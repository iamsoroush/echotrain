from .base_model import BaseModel

MAPPER = {'model_base': BaseModel}


def get_model_by_name(name):

    """
    Returns model class by name

    :param name: name of model class
    :return: a model of type BaseModel
    """

    return MAPPER[name]
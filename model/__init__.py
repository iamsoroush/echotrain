from .model_base import ModelBase

MAPPER = {'model_base': ModelBase}


def get_model_by_name(name):

    """
    Returns model class by name

    :param name: name of model class
    :return: a model of type BaseModel
    """

    return MAPPER[name]
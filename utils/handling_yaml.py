import yaml


def load_config_file(path):

    """
    loads the json config file and returns a dictionary

    :param path: path to json config file
    :return: a dictionary of {config_name: config_value}
    """

    with open(path) as f:
        # use safe_load instead load
        data_map = yaml.safe_load(f)

    config_obj = Struct(**data_map)
    return config_obj


class Struct:
    def __init__(self, **entries):
        for k, v in entries.items():
            if isinstance(v, dict):
                self.__dict__[k] = Struct(**v)
            else:
                self.__dict__[k] = v

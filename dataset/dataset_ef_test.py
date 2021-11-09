from .dataset_ef import EFDataset
from utils import handling_yaml
import numpy


def test_volume_dataset():
    config_path = 'D:\projects\AI_medic_projects\echotrain\config\config_ef.yaml'

    config_file = handling_yaml.load_config_file(config_path)

    ef_dataset = EFDataset(config_file)

    subset = 'val'
    input_type = 'image'

    assert subset in ('train', 'test', 'val'), 'pass either "test" or "validation" or "train" for "subset" ' \
                                               'argument. '
    assert input_type in ('image', 'label'), 'pass either "image"  or "label" for "type" argument.'

    x, y = ef_dataset.volume_dataset(input_type, subset)

    assert len(x.shape) == 4 and type(x) == numpy.ndarray and len(y.shape) == 2 and type(y) == numpy.ndarray


def test_ef_dataset():
    config_path = 'D:\projects\AI_medic_projects\echotrain\config\config_ef.yaml'

    config_file = handling_yaml.load_config_file(config_path)

    assert config_file is not None

    ef_dataset = EFDataset(config_file)

    subset = 'val'
    input_type = 'image'

    assert subset in ('train', 'test', 'val'), 'pass either "test" or "validation" or "train" for "subset" ' \
                                               'argument. '
    assert input_type in ('image', 'label'), 'pass either "image"  or "label" for "type" argument.'

    x, y = ef_dataset.ef_dataset(input_type, subset)

    assert len(x.shape) == 5 and type(x) == numpy.ndarray and len(y.shape) == 1 and type(y) == numpy.ndarray

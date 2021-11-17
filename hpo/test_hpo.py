from .hpo_baseline import HPOBaseline
from utils import load_config_file
from pydoc import locate
import os.path


def create_properties():
    config_path = '/content/echotrain/config/hpo_config.yaml'
    config_file = load_config_file(config_path)

    try:
        preprocessor_class_path = config_file.preprocessor_class
    except AttributeError:
        raise Exception('could not find preprocessor_class')

    try:
        dataset_class_path = config_file.dataset_class
    except AttributeError:
        raise Exception('could not find dataset_class')

    # Dataset
    print('preparing dataset ...')
    dataset_class = locate(dataset_class_path)
    dataset = dataset_class(config_file)
    train_data_gen, val_data_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

    # Preprocessor
    print('preparing pre-processor ...')
    preprocessor_class = locate({preprocessor_class_path})
    preprocessor = preprocessor_class(config_file)
    train_data_gen = preprocessor.add_preprocess(train_data_gen, True)
    val_data_gen = preprocessor.add_preprocess(val_data_gen, False)

    return train_data_gen, val_data_gen, n_iter_train, n_iter_val


def test_search_hp():
    config_path = '/content/echotrain/config/hpo_config.yaml'
    config_file = load_config_file(config_path)
    assert config_file is not None

    train_data_gen, val_data_gen, n_iter_train, n_iter_val = create_properties()

    assert str(type(train_data_gen)) == """<class 'generator'>""" and str(
        type(val_data_gen)) == """<class 'generator'>""" and type(n_iter_train) == int and type(n_iter_val) == int

    #     here we will change n_iter_train and n_iter_val to different number because of time
    hpo_baseline = HPOBaseline(config_file)

    tuner = hpo_baseline.search_hp(train_data_gen, val_data_gen, n_iter_train=2, n_iter_val=2)

    assert str((type(tuner))) == """<class 'keras_tuner.tuners.randomsearch.RandomSearch'>"""


def test_export_config():
    config_path = '/content/echotrain/config/hpo_config.yaml'
    config_file = load_config_file(config_path)
    assert config_file is not None

    train_data_gen, val_data_gen, n_iter_train, n_iter_val = create_properties()
    hpo_baseline = HPOBaseline(config_file)
    tuner = hpo_baseline.search_hp(train_data_gen, val_data_gen, n_iter_train=2, n_iter_val=2)
    assert str((type(tuner))) == """<class 'keras_tuner.tuners.randomsearch.RandomSearch'>""" and type(
        config_path) == str
    hpo_baseline.export_config(config_path, tuner)

    directory = config_file.hpo.directory

    assert os.path.exists(os.path.join(directory, "best_hp_config.yaml"))

"""
1- Put your config.yaml file into a directory:
    /experiments/unet_0/config.yaml
2- Run this file:
    python train.py --experiment_dir /experiments/unet_0

"""

import pathlib
import argparse
from pydoc import locate

from training import Trainer
from utils.handling_yaml import load_config_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_dir',
                        type=str,
                        help='directory of the config file',
                        required=True)

    return parser.parse_args()


def check(experiment_dir: pathlib.Path):
    if not experiment_dir.is_dir():
        raise Exception(f'{experiment_dir} is not a directory.')

    yaml_files = list(experiment_dir.glob('*.yaml'))
    if not any(yaml_files):
        raise Exception(f'no .yaml files found.')
    elif len(yaml_files) > 1:
        raise Exception(f'found more than one .yaml files.')

    return yaml_files[0]


if __name__ == '__main__':
    args = parse_args()

    experiment_dir = pathlib.Path(args.experiment_dir)
    config_path = check(experiment_dir)
    config_file = load_config_file(config_path.absolute())

    try:
        model_class_path = config_file.model_class
    except AttributeError:
        raise Exception('could not find model_class')

    try:
        preprocessor_class_path = config_file.preprocessor_class
    except AttributeError:
        raise Exception('could not find preprocessor_class')

    try:
        dataset_class_path = config_file.dataset_class
    except AttributeError:
        raise Exception('could not find dataset_class')

    dataset_class = locate(dataset_class_path)
    dataset = dataset_class(config_file)
    train_data_gen, val_data_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

    model_class = locate(model_class_path)
    model_obj = model_class(config_file)
    model = model_obj.generate_training_model()

    trainer = Trainer(base_dir=experiment_dir, config=config_file)

    history = trainer.train(model=model,
                            train_data_gen=train_data_gen,
                            val_data_gen=val_data_gen,
                            n_iter_train=n_iter_train,
                            n_iter_val=n_iter_val)

    trainer.export()

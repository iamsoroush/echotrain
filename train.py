"""
1- Put your config.yaml file into a directory:
    /experiments/unet_0/config.yaml
2- Run this file:
    python train.py --experiment_dir /experiments/unet_0

"""

import pathlib
import argparse

from dataset import get_dataset_by_name
from model import get_model_by_name
from training import TrainerBase
from utils.handling_yaml import load_config_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_dir',
                        type=str,
                        help='directory of the config file',
                        required=True)

    args = parser.parse_args()
    return args


def check(model_dir: pathlib.Path):
    if not model_dir.is_dir():
        raise Exception(f'{model_dir} is not a directory.')

    yaml_files = list(model_dir.glob('*.yaml'))
    if not any(yaml_files):
        raise Exception(f'no .yaml files found.')
    elif len(yaml_files) > 1:
        raise Exception(f'found two .yaml files.')

    return yaml_files[0]


if __name__ == '__main__':
    args = parse_args()

    model_dir = pathlib.Path(args.experiment_dir)
    config_path = check(model_dir)
    config_file = load_config_file(config_path.absolute())

    dataset = get_dataset_by_name(config_file.dataset_class_name)(config_file.data_handler)
    train_data_gen, val_data_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

    model_class = get_model_by_name(config_file.model_name)(config_file.model)
    model = model_class.generate_model()

    trainer = TrainerBase(config_file.trainer)

    trainer.train(model=model,
                  train_data_gen=train_data_gen,
                  val_data_gen=val_data_gen,
                  n_iter_train=n_iter_train,
                  n_iter_val=n_iter_val)

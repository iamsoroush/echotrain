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
from utils import load_config_file, check_for_config_file, setup_mlflow


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--experiment_dir',
                        type=str,
                        help='directory of the config file',
                        required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    experiment_dir = pathlib.Path(args.experiment_dir)
    config_path = check_for_config_file(experiment_dir)
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

    # Dataset
    dataset_class = locate(dataset_class_path)
    dataset = dataset_class(config_file)
    train_data_gen, val_data_gen, n_iter_train, n_iter_val = dataset.create_data_generators()

    # Preprocessor
    preprocessor_class = locate(preprocessor_class_path)
    preprocessor = preprocessor_class(config_file)
    train_data_gen = preprocessor.add_preprocess(train_data_gen)
    val_data_gen = preprocessor.add_preprocess(val_data_gen)

    # Model
    model_class = locate(model_class_path)
    model_obj = model_class(config_file)
    model = model_obj.generate_training_model()

    # Train
    mlflow_active_run = setup_mlflow(mlflow_tracking_uri=config_file.mlflow.tracking_uri,
                                     mlflow_experiment_name=config_file.mlflow.experiment_name,
                                     base_dir=experiment_dir)
    trainer = Trainer(base_dir=experiment_dir, config=config_file)

    print('training ...')
    history = trainer.train(model=model,
                            train_data_gen=train_data_gen,
                            val_data_gen=val_data_gen,
                            n_iter_train=n_iter_train,
                            n_iter_val=n_iter_val,
                            active_run=mlflow_active_run)

    exported_dir = trainer.export()
    print(f'exported to {exported_dir}.')

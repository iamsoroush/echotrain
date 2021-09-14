import argparse

from dataset import get_dataset_by_name
from model import get_model_by_name
from training import TrainerBase
from utils import load_config_file


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir',
                        type=str,
                        help='directory of dataset',
                        required=True)

    parser.add_argument('--checkpoints_dir',
                        type=str,
                        help='directory of checkpoints',
                        required=True)

    parser.add_argument('--logs_dir',
                        type=str,
                        help='directory for tensorboard logs',
                        required=True)

    parser.add_argument('--config_path',
                        type=str,
                        help='path to config file',
                        required=True)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config_file = load_config_file(args.config_path)
    dataset_dir = args.dataset_dir
    checkpoints_dir = args.checkpoints_dir
    logs_dir = args.logs_dir

    dataset = get_dataset_by_name(config_file.dataset_class_name)(config_file.batch_size,
                                                                  config_file.input_res,
                                                                  config_file)
    model_class = get_model_by_name(config_file.model_name)(config_file)
    trainer = TrainerBase(checkpoints_dir=checkpoints_dir, logs_dir=logs_dir, config=config_file)

    train_data_gen, val_data_gen, n_iter_train, n_iter_val = dataset.create_data_generators(dataset_dir=dataset_dir)
    model = model_class.generate_model()

    trainer.train(model=model,
                  train_data_gen=train_data_gen,
                  val_data_gen=val_data_gen,
                  n_iter_train=n_iter_train,
                  n_iter_val=n_iter_val)

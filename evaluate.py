import sys
sys.path.append('echotrain')

import pathlib
import argparse

import pandas as pd

from evaluation import Evaluator


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
    exported_dir = experiment_dir.joinpath('exported')

    evaluator = Evaluator(exported_dir=exported_dir)
    eval_report, val_df = evaluator.generate_report()
    summary_report = eval_report.describe()

    merged_df = pd.merge(val_df, eval_report, left_index=True, right_index=True)
    merged_df.to_csv(experiment_dir.joinpath('evaluation_report.csv'))
    summary_report.to_csv(experiment_dir.joinpath('summary_report.csv'))

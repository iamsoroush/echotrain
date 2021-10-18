import sys
sys.path.append('echotrain')

import pathlib
import argparse

from evaluator.evaluator import Evaluator


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

    eval = Evaluator()
    eval_report = eval.generate_report(exported_dir)
    summary_report = eval_report.describe()

    eval_report.to_csv(experiment_dir.joinpath('eval_report.csv'))
    summary_report.to_csv(experiment_dir.joinpath('summary_report.csv'))

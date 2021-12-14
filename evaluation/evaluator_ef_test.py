from utils import handling_yaml
from .evaluator_ef import EFEvaluator
import pandas
import os.path


def test_evaluate():
    config_path = 'D:\projects\AI_medic_projects\echotrain\config\config_ef.yaml'

    config_file = handling_yaml.load_config_file(config_path)

    assert config_file is not None

    ef_evaluator = EFEvaluator(config_file)
    subset = 'val'
    assert subset in ('train', 'test', 'val'), 'pass either "test" or "validation" or "train" for "subset" argument. '
    ef_dataframe = ef_evaluator.evaluate(subset)
    assert type(ef_dataframe) == pandas.core.frame.DataFrame

    exported_dir = config_file.exported_dir

    assert os.path.exists(os.path.join(exported_dir, "exported", "dataframe_of_evaluation.csv"))

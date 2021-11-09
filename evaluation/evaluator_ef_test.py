from utils import handling_yaml
from .evaluator_ef import EFEvaluator


def test_evaluate():
    config_path = 'D:\projects\AI_medic_projects\echotrain\config\config_ef.yaml'

    config_file = handling_yaml.load_config_file(config_path)

    assert config_file is not None

    ef_evaluator = EFEvaluator(config_file)
    subset = 'val'
    ef_dataframe = ef_evaluator.evaluate(subset)

    assert type(ef_evaluator) ==

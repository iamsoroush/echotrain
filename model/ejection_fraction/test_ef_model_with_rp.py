from utils import handling_yaml
from .ef_model_with_rp import EFModel_RP
import numpy
import os.path


def test_ef_estimation():
    config_path = 'D:\projects\AI_medic_projects\echotrain\config\config_ef.yaml'

    config_file = handling_yaml.load_config_file(config_path)

    assert config_file is not None

    ef_model_encoder = EFModel_RP(config_file)

    ed_frame = numpy.random.randn(256, 256, 1)
    es_frame = numpy.random.randn(256, 256, 1)

    assert len(ed_frame.shape) == 3 and type(ed_frame) == numpy.ndarray and len(es_frame.shape) == 3 and type(
        es_frame) == numpy.ndarray

    ef_estimation = ef_model_encoder.ef_estimation(ed_frame, es_frame)

    assert type(ef_estimation) == float


def test_train():
    config_path = 'D:\projects\AI_medic_projects\echotrain\config\config_ef.yaml'

    config_file = handling_yaml.load_config_file(config_path)

    assert config_file is not None

    ef_model_encoder = EFModel_RP(config_file)

    model = ef_model_encoder.train()

    ef_model_encoder.export(model)

    exported_dir = config_file.exported_dir

    assert os.path.exists(os.path.join(exported_dir, "exported", "rp_to_v.sav"))

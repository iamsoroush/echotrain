from ejection_fraction.ejection_fraction_estimation import EFEstimation
from ejection_fraction.cycle_detection import CycleDetector
from glob import glob
import os


class EFInference:
    """
    This class acts as and inference engine for ejection fraction (EF) estimation which contains:
        1. cycle detection
        2. volume estimation
        3. ejection fraction estimation

    How To:
        ef_engine = EFInference(ef_experiment_directory)
        ed_frame, es_frame, ed_idx, es_idx = ef_engine.detect_cycle(video_label_frames)
        ed_volume, es_volume, ef = ef_engine.estimate_ef(ed_frame, es_frame)
    """
    def __init__(self, ef_exp_dir):
        """
        :param ef_exp_dir: ef experiments directory which contains the volume estimation trained models, string
        """
        self.ef_exp_dir = ef_exp_dir

    @staticmethod
    def detect_cycle(video_label_frames):
        """
        detecting cycle of a video by the segmented frame area
        :param video_label_frames: list of segmented frames, list with dtype of numpy.array
        :return ed_frame: end diastole frame, numpy.array
        :return es_frame: end systole frame, numpy.array
        :return ed_idx: end diastole frame number, int
        :return es_idx: end systole frame, int
        """

        cycle_detector = CycleDetector(video_label_frames)
        ed_frame, es_frame, ed_idx, es_idx = cycle_detector.detect_ed_es()

        return ed_frame, es_frame, ed_idx, es_idx

    def estimate_ef(self, ed_frame, es_frame):
        """
        volume estimation of the ed and es frames and ef estimation
        :param ed_frame: end diastole frame, numpy.array
        :param es_frame: end systole frame, numpy.array
        :return ed_volume: end diastole volume, float
        :return es_volume: end systole volume, float
        :return ef: ejection fraction estimation, float
        """
        ef_est_class = EFEstimation()
        model = ef_est_class.load_model(glob(os.path.join(self.ef_exp_dir, '*'))[0].replace('\\', '/'))
        ed_volume = ef_est_class.volume_estimation(frame=ed_frame, model=model)
        es_volume = ef_est_class.volume_estimation(frame=es_frame, model=model)
        ef = ef_est_class.ef_estimation(es_frame=es_frame, ed_frame=ed_frame, model=model)

        return ed_volume, es_volume, ef
import warnings
import cv2
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks, argrelextrema


class CycleDetector:
    """
    A class for cycle detection of a segmented video label frames

    How To:
    cycle_detector = CycleDetector(video_label_frames)
    ed_frame, es_frame, ed_idx, es_idx = cycle_detector.detect_ed_es()
    """
    def __init__(self, label_video_frames, method='radius'):
        """

        :param label_video_frames: list of segmented frames, list with dtype of numpy.array
        :param method: name cycle detection methods ( radius, topological, argrelextrema), string
        :param frame_area: list of the segemented area estimated with cv2.CountNonZero module, list of int
        :param cycles: list of cycles found in the self.find_cycles() module, list of tuples of int
        """
        self.label_video_frames = label_video_frames
        self.method = method
        self.frames_area = [cv2.countNonZero(frame) for frame in self.label_video_frames]
        self.cycles = self.find_cycles()

    def find_peaks(self, **kwargs):
        """
        finds peaks and valleys with the method and parameters given
        :param kwargs: radius --> 1. confidence, int ( default value == 20 )
                       topological --> 1. width, int ( no default values )
                                       2. prominence, int ( default value == 50 )
                                       ...
                       argrelextrema --> ...
        :return: min peaks: as valleys, list of int
        :return: max peaks: as peaks, list of int
        """
        min_peaks = []
        max_peaks = []

        volumes = np.array(self.frames_area)
        inv_volume = volumes.max() - volumes

        gs_volumes = gaussian_filter1d(volumes, sigma=3)
        gs_inv_volumes = gaussian_filter1d(inv_volume, sigma=3)
        if self.method == 'radius':
            if kwargs:
                try:
                    confidence = kwargs['confidence']
                except KeyError:
                    raise KeyError("The parameter for radius method is 'confidence' which is not provided!!!")
            else:
                warnings.warn('Using default value(s) for the parameter(s)')
                confidence = 20
            num_frames = len(volumes)
            for i in range(num_frames):
                min_flag = True
                max_flag = True
                for j in range(1, confidence + 1):
                    if i + j < num_frames:
                        if not (volumes[i] < volumes[i + j]):
                            min_flag = False
                        if not (volumes[i] > volumes[i + j]):
                            max_flag = False
                    if i - j >= 0:
                        if not (volumes[i] < volumes[i - j]):
                            min_flag = False
                        if not (volumes[i] > volumes[i - j]):
                            max_flag = False
                if min_flag:
                    min_peaks.append(i)
                elif max_flag:
                    max_peaks.append(i)

        if self.method == 'topological':
            if not kwargs:
                kwargs['prominence'] = 50

            max_peaks, _ = find_peaks(gs_volumes, **kwargs)
            min_peaks, _ = find_peaks(gs_inv_volumes, **kwargs)

        if self.method == 'argrelextrema':

            max_peaks = argrelextrema(gs_volumes, np.greater, **kwargs)[0]
            min_peaks = argrelextrema(gs_volumes, np.less, **kwargs)[0]

        return min_peaks, max_peaks

    def find_cycles(self, **kwargs):
        """
        finds cycles from min_praks and max_peaks
        :param kwargs: same as self.find_peaks() module
        :return: cycles: list of cycles, list of tuples of int
        """
        min_peaks, max_peaks = self.find_peaks(**kwargs)
        # assert (min_peaks != [] or max_peaks != []), f'there are no min_peak and max_peak'
        # assert min_peaks != [], f'there is no min_peak'
        # assert max_peaks != [], f'there is no max_peak'

        ed = max_peaks[0]
        es = min_peaks[0]

        if ed < es:
            cycles = [(ed_, es_) for ed_, es_ in zip(max_peaks, min_peaks)]
        else:
            init_idx = 1
            for idx in range(init_idx, len(min_peaks)):
                es = min_peaks[idx]
                if es > ed:
                    init_idx = idx
                    break
            cycles = [(ed_, es_) for ed_, es_ in zip(max_peaks, min_peaks[init_idx:])]

        self.cycles = cycles
        return cycles

    def detect_ed_es(self):
        """
        chooses a cycle and returns the ed ad es frames
        :return ed_frame: end diastole frame, numpy.array
        :return es_frame: end systole frame, numpy.array
        :return ed_idx: end diastole frame number, int
        :return es_idx: end systole frame, int
        """
        cycles = self.cycles

        distances = np.array([(self.frames_area[cycle[0]] - self.frames_area[cycle[1]]) for cycle in cycles])

        cycle = cycles[np.where(distances == distances.max())[0][0]]
        ed = cycle[0]
        es = cycle[1]
        ed_frame = self.label_video_frames[ed]
        es_frame = self.label_video_frames[es]
        return ed_frame, es_frame, ed, es

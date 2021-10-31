import cv2


class CycleDetector:
    """

    """
    def __init__(self, label_video_frames):
        """

        :param label_video_frames:
        """
        self.label_video_frames = label_video_frames
        self.frames_volume = [cv2.countNonZero(frame) for frame in self.label_video_frames]

    def _find_cycle(self, confidence):
        """

        :param confidence:
        :return:
        """
        min_peaks = []
        max_peaks = []
        num_frames = len(self.frames_volume)
        for i in range(num_frames):
            min_flag = True
            max_flag = True
            for j in range(1, confidence + 1):
                if i+j < num_frames:
                    if not (self.frames_volume[i] < self.frames_volume[i + j]):
                        min_flag = False
                    if not (self.frames_volume[i] > self.frames_volume[i + j]):
                        max_flag = False
                if i-j >= 0:
                    if not (self.frames_volume[i] < self.frames_volume[i - j]):
                        min_flag = False
                    if not (self.frames_volume[i] > self.frames_volume[i - j]):
                        max_flag = False
            if min_flag:
                min_peaks.append((self.frames_volume[i], i))
            elif max_flag:
                max_peaks.append((self.frames_volume[i], i))

        return min_peaks, max_peaks

    def detect_ed_es(self, confidence=20):
        """

        :param confidence:
        :return:
        """
        min_peaks, max_peaks = self._find_cycle(confidence)
        print(max_peaks)
        print(min_peaks)
        assert (min_peaks != [] or max_peaks != []), f'there are no min_peak and max_peak with {confidence} confidence'
        assert min_peaks != [], f'there is no min_peak with {confidence} confidence'
        assert max_peaks != [], f'there is no max_peak with {confidence} confidence'
        ed = max_peaks[0]
        es = min_peaks[0]
        if ed[1] > min_peaks[-1][1]:
            if ed[1] < es[1]:
                for idx in range(1, len(max_peaks)):
                    ed = max_peaks[idx]
                    if ed[1] > es[1]:
                        break
                assert ed[1] > es[1], 'there is no ED frame after selected ES frame!'
        else:
            if ed[1] > es[1]:
                for idx in range(1, len(min_peaks)):
                    es = min_peaks[idx]
                    if ed[1] < es[1]:
                        break
                assert ed[1] < es[1], 'there is no ES frame after selected ED frame!'

        ed_frame = self.label_video_frames[ed[1]]
        es_frame = self.label_video_frames[es[1]]
        return ed_frame, es_frame, ed, es


# def read_video(video_dir, to_gray=True):
#
#     """
#     loads the video from the directory given
#     :param video_dir: video address directory, str
#     :param to_gray: makes the frames gray-scale if needed, bool
#     :return: video_frames: video frames array, np.array
#     :return: video_info: list of video information ( frame count, fps, duration ), list
#     """
#
#     # a list to be used for storing the frames
#     video_frames = []
#
#     # capturing the video with OpenCV lib.
#     vidcap = cv2.VideoCapture(video_dir)
#
#     frame_count = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
#     fps = vidcap.get(cv2.CAP_PROP_FPS)
#     duration = frame_count / fps
#
#     # counting the frames
#     count = 0
#     while vidcap.isOpened():
#         success, frame = vidcap.read()
#         if success:
#             if to_gray:
#                 # converting the frames from rgb to gray-scale ( just for compression )
#                 frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             # frame_label = self.process(frame)
#             video_frames.append(frame)
#             # video_labels.append(frame_label)
#         else:
#             break
#         count += 1
#
#     # releasing the video capture object
#     vidcap.release()
#
#     # validating the frame count
#     if count != frame_count:
#         frame_count = count
#
#     video_info = [frame_count, fps, duration]
#     return np.array(video_frames), video_info  # , video_labels

# if __name__ == '__main__':
#     video_dir = 'D:/AIMedic/FinalProject_echocardiogram/echoC_Dataset/test/0X1002E8FBACD08477_label.mp4'
#     video_frames, video_info = read_video(video_dir)
#     print(video_frames.shape)
#     es_ed_detection = CycleDetector(video_frames)
#     ed_frame, es_frame, ed_info, es_info = es_ed_detection.detect_ed_es()
#     print(ed_info)
#     print(es_info)

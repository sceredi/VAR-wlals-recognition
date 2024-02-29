import numpy as np

from handcrafted.app.dataset.video import Video
from handcrafted.app.plotter.framesPlotter import FramesPlotter
from wlasl_mediapipe.app.mp.hands_extractor import MediapipeHandsExtractor
from wlasl_mediapipe.app.mp.models.sign_model import SignModel
from wlasl_mediapipe.app.utils.mp.helper.hand_landmark_analyzer import analyze_landmarks


class MediapipeVideo:
    def __init__(self, video: Video, plot: bool = True):
        self.video = video
        self.model = MediapipeHandsExtractor()
        self.sign_model = self._load_sign_model(plot=plot)

    def get_base_video(self):
        return self.video


    def _load_sign_model(self, plot: bool = True) -> SignModel:
        results, annotated_frames = self.model.process_video(self.video)
        if plot:
            FramesPlotter(annotated_frames, to_rgb=False).plot_grid()
        left_hand_list = []
        right_hand_list = []
        for result in results:
            left_hand, right_hand = analyze_landmarks(result)
            if len(left_hand) == 0:
                left_hand = np.nan_to_num(np.zeros((21, 3)))
            if len(right_hand) == 0:
                right_hand = np.nan_to_num(np.zeros((21, 3)))
            left_hand_list.append(np.nan_to_num(left_hand).reshape(63).tolist())
            right_hand_list.append(np.nan_to_num(right_hand).reshape(63).tolist())
        return SignModel(left_hand_list, right_hand_list)

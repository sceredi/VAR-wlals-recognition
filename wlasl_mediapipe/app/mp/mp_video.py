from typing import List

from handcrafted.app.dataset.video import Video
from handcrafted.app.plotter.framesPlotter import FramesPlotter
from wlasl_mediapipe.app.mp.hands_extractor import MediapipeHandsExtractor
from wlasl_mediapipe.app.mp.models.sign_model import SignModel
from wlasl_mediapipe.app.utils.mp.helper.hand_landmark_analyzer import analyze_landmarks


class MediapipeVideo:
    def __init__(self, video: Video):
        self.video = video
        self.model = MediapipeHandsExtractor()
        self.pose = None

    def get_base_video(self):
        return self.video

    def get_poses(self, plot: bool = True) -> List[SignModel]:
        if self.pose is None:
            self.pose = self._load_poses(plot=plot)
        return self.pose

    def _load_poses(self, plot: bool = True) -> List[SignModel]:
        results, annotated_frames = self.model.process_video(self.video)
        if plot:
            FramesPlotter(annotated_frames, to_rgb=False).plot_grid()
        for result in results:
            left_hand_list, right_hand_list = analyze_landmarks(result)
            print(f"Left hand: {left_hand_list}")
            print(f"Right hand: {right_hand_list}")
        raise NotImplementedError


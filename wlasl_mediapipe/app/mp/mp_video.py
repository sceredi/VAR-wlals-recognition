import os

from handcrafted.app.dataset.video import Video
from handcrafted.app.plotter.framesPlotter import FramesPlotter
from wlasl_mediapipe.app.mp.hands_extractor import MediapipeLandmarksExtractor
from wlasl_mediapipe.app.mp.models.sign_model import SignModel
from wlasl_mediapipe.app.utils.mp.file_utils import save_array
from wlasl_mediapipe.app.utils.mp.helper.hand_landmark_analyzer import extract_landmarks


class MediapipeVideo:
    def __init__(self, video: Video, plot: bool = True):
        self.video = video
        if not os.path.exists(f"data/mp/{self.video.video_id}"):
            self.model = MediapipeLandmarksExtractor()
            self.sign_model = self._load_sign_model(plot=plot)

    def get_base_video(self):
        return self.video


    def _load_sign_model(self, plot: bool = True) -> SignModel:
        if os.path.exists(f"data/mp/{self.video.video_id}"):
            print(f"Path exists: data/mp/{self.video.video_id}")
            return SignModel.load(self.video.video_id)
        else:
            print(f"Path does not exist: data/mp/{self.video.video_id}")
            results, annotated_frames = self.model.process_video(self.video)
            if plot:
                FramesPlotter(annotated_frames, to_rgb=False).plot_grid()
            pose_list = []
            left_hand_list = []
            right_hand_list = []
            for result in results:
                pose, left_hand, right_hand = extract_landmarks(result)
                pose_list.append(pose)
                left_hand_list.append(left_hand)
                right_hand_list.append(right_hand)
            os.mkdir(f"data/mp/{self.video.video_id}")
            save_array(pose_list, f"data/mp/{self.video.video_id}/pose_{self.video.video_id}.pickle")
            save_array(left_hand_list, f"data/mp/{self.video.video_id}/lh_{self.video.video_id}.pickle")
            save_array(right_hand_list, f"data/mp/{self.video.video_id}/rh_{self.video.video_id}.pickle")
            return SignModel(left_hand_list, right_hand_list)

from handcrafted.app.dataset.video import Video
from handcrafted.app.plotter.framesPlotter import FramesPlotter
from wlasl_mediapipe.app.mp.hands_extractor import MediapipeHandsExtractor
from wlasl_mediapipe.app.mp.models.sign_model import SignModel


class MediapipeVideo:
    def __init__(self, video: Video):
        self.video = video
        self.model = MediapipeHandsExtractor()
        self.pose = None

    def get_base_video(self):
        return self.video

    def get_pose(self, plot: bool = True) -> SignModel:
        if self.pose is None:
            self.pose = self._load_pose(plot=plot)
        return self.pose

    def _load_pose(self, plot: bool = True) -> SignModel:
        results, annotated_frames = self.model.process_video(self.video)
        if plot:
            FramesPlotter(annotated_frames, to_rgb=False).plot_grid()
        for result in results:
            print(result)
            input("Press Enter to continue...")
            # if result is not None:
            #     return SignModel(result)
        raise NotImplementedError

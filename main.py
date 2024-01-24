from app.dataset.dataset import Dataset
from app.dataset.video import Video
from app.edge.detector import EdgeDetector
from app.plotter.framesPlotter import FramesPlotter
from app.roi.extractor import RoiExtractor
from app.flow.calculator import FlowCalculator


def plot_video(video: Video):
    roi_extractor = RoiExtractor(video.get_frames(), video.bbox)
    roi_frames = roi_extractor.extract(remove_background=False)
    edge_detector = EdgeDetector(roi_frames)
    edge_frames = edge_detector.detect()
    plotter = FramesPlotter(edge_frames)
    # plotter.plot_grid()
    flow_calculator = FlowCalculator(roi_frames)
    flow_frames = flow_calculator.calculate()
    plotter = FramesPlotter(flow_frames)
    plotter.plot_grid()


if __name__ == "__main__":
    dataset = Dataset("data/WLASL_v0.3.json")
    for video in dataset.videos:
        print("Plotting video: ", video.get_path())
        plot_video(video)
    # plot_video(dataset.videos[0])

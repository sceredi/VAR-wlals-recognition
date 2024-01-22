import matplotlib.pyplot as plt
from app.dataset.dataset import Dataset
from app.dataset.video import Video
from app.plotter.framesPlotter import FramesPlotter
from app.utilities.preprocess import preprocess

def plot_video(video: Video):
    frames = video.get_bounded_frames()
    plotter = FramesPlotter(frames)
    plotter.plot()


if __name__ == '__main__':
    fg = preprocess('data/videos/00336.mp4')
    plt.imshow(fg)
    plt.axis('off')
    plt.show()
    dataset = Dataset('data/WLASL_v0.3.json')
    print(dataset.videos)
    plot_video(dataset.videos[0])


import math
import os

import numpy as np
from tqdm import tqdm


class DatasetLoader:
    def __init__(
        self,
        directory: str = "./data/frames_no_bg/",
    ) -> None:
        self._path = directory
        self.signers = self._load_dataset()
        self.num_signers = len(self.signers)

    def _load_dataset(self):
        signers = dict()
        for dirname, _, filenames in tqdm(os.walk(self._path)):
            for filename in filenames:
                dirname = dirname.removeprefix(self._path)
                dirname = os.path.normpath(dirname)
                signer_id, video_id = dirname.split(os.sep)
                signer_id = signer_id
                video_id = video_id
                if signer_id not in signers:
                    signer = Signer(signer_id)
                    signers[signer_id] = signer
                else:
                    signer = signers[signer_id]
                existing_video = next(
                    (v for v in signer.videos if v.id == video_id), None
                )
                if existing_video is None:
                    video = Video(video_id)
                    signer.videos.append(video)
                else:
                    video = existing_video
                frame_id = filename.removesuffix(".png")
                frame = Frame(
                    frame_id, os.path.join(os.path.normpath(self._path), dirname, filename)
                )
                video.frames.append(frame)
        return signers


class Signer:
    def __init__(self, id: str) -> None:
        self.id = id
        self.videos: list[Video] = []

    def split(
        self, val_split: float, test_split: float, random_state: int = 42
    ):
        np.random.seed(random_state)
        val_videos = []
        test_videos = []
        test_videos_to_extract = math.ceil(len(self.videos) * test_split)
        test_videos_ids = np.random.choice(
            np.arange(0, len(self.videos)),
            test_videos_to_extract,
            replace=False,
        ).tolist()
        for test_video_id in test_videos_ids:
            test_videos.append(self.videos[test_video_id])
        for i, test_video_id in enumerate(test_videos_ids):
            self.videos.remove(self.videos[test_video_id - i])
        val_videos_to_extract = math.ceil(len(self.videos) * val_split)
        val_videos_ids = np.random.choice(
            np.arange(0, len(self.videos)),
            val_videos_to_extract,
            replace=False,
        ).tolist()
        for val_video_id in val_videos_ids:
            val_videos.append(self.videos[val_video_id])
        for i, val_video_id in enumerate(val_videos_ids):
            self.videos.remove(self.videos[val_video_id - i])
        return self.videos, val_videos, test_videos


class Video:
    def __init__(self, id: str) -> None:
        self.id = id
        self.frames: list[Frame] = []

    def __str__(self):
        return f"Video {self.id}"

    def __eq__(self, other) -> bool:
        return self.id == other.id

    def extract_frames(self, frames_split: float, seed: int):
        frames = []
        frames_to_extract = math.ceil(len(self.frames) * frames_split)
        np.random.seed(seed)
        frame_ids = np.random.choice(
            np.arange(0, len(self.frames)), frames_to_extract, replace=False
        ).tolist()
        for frame_id in frame_ids:
            frames.append(self.frames[frame_id])
        return frames


class Frame:
    def __init__(self, id: str, path: str) -> None:
        self.id = id
        self.path = path

    def __str__(self):
        return f"Frame {self.id}, path: {self.path}"

    def load_frame(self):
        pass

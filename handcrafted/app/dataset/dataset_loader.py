import os


class DatasetLoader:
    def __init__(
        self,
        directory: str = "./data/frames_no_bg/",
        val_split: float = 0.2,
        test_split: float = 0.2,
        frames_split: float = 0.1,
    ) -> None:
        self._path = directory
        self._val_split = val_split
        self._test_split = test_split
        self._frames_split = frames_split
        self.signers = self._load_dataset()

    def _load_dataset(self):
        signers = dict()
        for dirname, _, filenames in os.walk(self._path):
            for filename in filenames:
                dirname = dirname.removeprefix(self._path)
                signer_id, video_id = dirname.split("/")
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
                    frame_id, os.path.join(self._path, dirname, filename)
                )
                video.frames.append(frame)
        return signers


class Signer:
    def __init__(self, id: str) -> None:
        self.id = id
        self.videos: list[Video] = []


class Video:
    def __init__(self, id: str) -> None:
        self.id = id
        self.frames: list[Frame] = []

    def __str__(self):
        return f"Video {self.id}"

    def __eq__(self, other) -> bool:
        return self.id == other.id


class Frame:
    def __init__(self, id: str, path: str) -> None:
        self.id = id
        self.path = path

    def __str__(self):
        return f"Frame {self.id}, path: {self.path}"

    def load_frame(self):
        pass

import json
from collections.abc import Callable
from typing import List, Tuple

from handcrafted.app.dataset.video import Video


class Dataset:
    def __init__(self, filename: str, only_keypoints=False, only_samples=False) -> None:
        self.filename = filename
        self._only_samples = only_samples
        if not only_keypoints:
            self.videos, self.glosses = self.load_videos()
        else:
            self.videos, self.glosses = self.load_videos_with_keypoints()

    def load_videos(self) -> Tuple[List[Video], List[str]]:
        with open(self.filename) as f:
            data = json.load(f)
        ret = []
        glosses = []
        for word_data in data:
            gloss = word_data["gloss"]
            glosses.append(gloss)
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(gloss, instance, self._only_samples)
                if not video.is_missing():
                    ret.append(video)
        return ret, glosses

    def load_videos_with_keypoints(self) -> Tuple[List[Video], List[str]]:
        with open(self.filename) as f:
            data = json.load(f)
        ret = []
        glosses = []
        for word_data in data:
            gloss = word_data["gloss"]
            glosses.append(gloss)
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(gloss, instance)
                if video.has_keypoints():
                    ret.append(video)
        return ret, glosses

    def update_videos(self) -> None:
        with open(self.filename) as f:
            data = json.load(f)
        for word_data in data:
            gloss = word_data["gloss"]
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(gloss, instance, self._only_samples)
                if video.is_missing() == False:
                    instance["frame_end"] = video.get_end()
                else:
                    instance = None
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)

    def get_videos(self, condition: Callable[[Video], bool]) -> List[Video]:
        return [video for video in self.videos if condition(video)]

    def __str__(self) -> str:
        return f"Dataset(filename={self.filename}, videos={self.videos})\n"

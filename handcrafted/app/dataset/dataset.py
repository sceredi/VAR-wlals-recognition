import json
from typing import List, Tuple

from handcrafted.app.dataset.video import Video


class Dataset:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.videos, self.glosses = self.load_videos()

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
                video = Video.from_instance(gloss, instance)
                if video.is_missing() == False:
                    ret.append(video)
        return ret, glosses

    def update_videos(self) -> None:
        with open(self.filename) as f:
            data = json.load(f)
        for word_data in data:
            gloss = word_data["gloss"]
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(gloss, instance)
                if video.is_missing() == False:
                    instance["frame_end"] = video.get_end()
                else:
                    instance = None
        with open(self.filename, "w") as f:
            json.dump(data, f, indent=4)


    def __str__(self) -> str:
        return f"Dataset(filename={self.filename}, videos={self.videos})\n"

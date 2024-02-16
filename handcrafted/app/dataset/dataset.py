import json
from typing import List

from handcrafted.app.dataset.video import Video


class Dataset:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.videos = self.load_videos()

    def load_videos(self) -> List[Video]:
        with open(self.filename) as f:
            data = json.load(f)
        ret = []
        for word_data in data:
            gloss = word_data["gloss"]
            instances = word_data["instances"]
            for instance in instances:
                video = Video.from_instance(gloss, instance)
                if video.is_missing() == False:
                    ret.append(video)
        return ret

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

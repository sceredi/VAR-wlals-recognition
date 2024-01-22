import json
from typing import List

from app.dataset.video import Video
class Dataset:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.videos = self.load_videos()

    def load_videos(self) -> List[Video]:
        with open(self.filename) as f:
            data = json.load(f)
        ret = []
        for word_data in data:
            gloss = word_data['gloss']
            instances = word_data['instances']
            for instance in instances:
                video = Video.from_instance(gloss, instance)
                if video.is_missing() == False:
                    ret.append(video)
        return ret
    
    def __str__(self) -> str:
        return f"Dataset(filename={self.filename}, videos={self.videos})\n"


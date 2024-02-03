import joblib
import os

from app.extractor.hog_extractor import HOGExtractor
from app.roi.extractor import RoiExtractor


class FeatureExtractor:
    def __init__(self, dataset, glosses):
        self.dataset = dataset
        self.glosses = glosses
        self.all_features = {}

    def flatten_frames(self, frames):
        new_frames = [frame.flatten() for frame in frames]
        return new_frames

    def extract_features(self, video):
        frames = video.get_frames()
        hog_frames = HOGExtractor(frames).process_frames()
        hog_sequence = self.flatten_frames(hog_frames)
        # skin_sequence = self.flatten_frames(get_skin_frames(get_roi_frames(video)))

        return {
            "hog_sequence": hog_sequence,
            # "skin_sequence": skin_sequence,
            # altre feature
        }

    def extract_and_save_all_features(self, output_filename):
        for video in self.dataset.videos:
            if video.video_id != "69241":
                break
            print(f"Processing video: {video.get_path()}")
            features = self.extract_features(video)
            self.all_features[video.video_id] = features

        self.save_features(output_filename)

    def save_features(self, filename):
        joblib.dump(self.all_features, filename)

    def load_features(self, filename):
        return joblib.load(filename)

    def delete_features(self, filename):
        if os.path.exists(filename):
            os.remove(filename)
            print(f"Il file {filename} è stato eliminato con successo.")
        else:
            print(f"Il file {filename} non esiste.")

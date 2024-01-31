# Purpose: Detects hands in a video stream

from typing import List
import cv2
import numpy as np

from app.dataset.frames import Frame, Frames


# TODO: find only best detection
class HaarDetector:
    def __init__(
        self,
        frames: List["np.ndarray"],
        classifier,
        detections_to_keep: int = 1,
    ):
        self.classifier = classifier
        self.frames = Frames(frames)
        self.detections_to_keep = detections_to_keep

    def _frame_detect(self, frame: Frame):
        gray_hist = cv2.equalizeHist(frame.gray)
        detections = self.classifier.detectMultiScale(gray_hist)
        detections = sorted(detections, key=lambda x: x[-1], reverse=True)
        return detections

    def detect(self):
        rects = []
        drawn_frames = []
        for frame in self.frames:
            detections = self._frame_detect(frame)
            rects.append(detections)
            frame = frame.rgb.copy()
            drawn_frames.append(frame)
            for i, rect in enumerate(detections):
                if i >= self.detections_to_keep:
                    break
                if rect is not None:
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return drawn_frames, rects

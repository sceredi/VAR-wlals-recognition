# Purpose: Detects hands in a video stream

from typing import List
import cv2
import numpy as np

from app.dataset.frames import Frame, Frames


class HandsDetector:
    def __init__(
        self,
        frames: List["np.ndarray"],
        classifier,
    ):
        self.classifier = classifier
        self.frames = Frames(frames)

    def _frame_detect(self, frame: Frame):
        gray_hist = cv2.equalizeHist(frame.gray)
        return self.classifier.detectMultiScale(gray_hist, scaleFactor=1.1, minNeighbors=5)

    def detect(self):
        rects = []
        hands = []
        for frame in self.frames:
            rect = self._frame_detect(frame)
            rects.append(rect)
            frame = frame.rgb.copy()
            hands.append(frame)
            for x, y, w, h in rect:
                print(x, y, w, h)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return hands, rects

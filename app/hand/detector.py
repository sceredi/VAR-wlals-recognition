# Purpose: Detects hands in a video stream

from typing import List
import cv2
import numpy as np

from app.dataset.frames import Frame, Frames


class HandsDetector:
    def __init__(
        self,
        frames: List["np.ndarray"],
        classifier=cv2.CascadeClassifier("app/hand/haarcascades/hand.xml"),
        params=dict(
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        ),
    ):
        self.frames = Frames(frames)
        self.classifier = classifier
        self.params = params

    def _frame_detect(self, frame: Frame):
        return self.classifier.detectMultiScale(frame.gray, **self.params)

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

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

    def avg_std_area_sorting_criteria(self, detection, avg_area, std_area):
        penalty_factor = 0.4
        # Absolute difference between the detection area and the average area
        detection_area = detection[2] * detection[3]
        area_difference = abs(detection_area - avg_area)
        penalized_difference = penalty_factor * area_difference * (1 + std_area)
        return penalized_difference

    def _frame_detect(self, frame: Frame):
        gray_hist = cv2.equalizeHist(frame.gray)
        detections = self.classifier.detectMultiScale(gray_hist)

        areas = [det[2] * det[3] for det in detections]
        avg_area = np.mean(areas)
        std_area = np.std(areas)
        print("Average area", avg_area)
        # x[-1]: ordina le detections in base alla loro confidenza
        # self.custom_sorting_criteria(x, avg_area): ordina le detections in base all'area media
        # x[2] * x[3]: ordina le detections in base all'area
        # x[2] / x[3]: ordina le detections in base all'aspect ratio
        # detections = sorted(detections, key=lambda x: (self.avg_std_area_sorting_criteria(x, avg_area, std_area), x[-1], x[2] * x[3]), reverse=True)
        detections = sorted(
            detections,
            key=lambda x: (self.avg_std_area_sorting_criteria(x, avg_area, std_area)),
            reverse=True,
        )

        # restituisce solo la migliore
        if detections:
            best_detection = detections[0]
            print("Best detection found", best_detection)
            print("Confidences:", detections[-1])
            print("Detections:", detections)
            return [best_detection]
        else:
            print("No detections found")
            return []

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

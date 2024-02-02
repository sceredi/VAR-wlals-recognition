# Purpose: Detects hands in a video stream

from typing import List
import cv2
import numpy as np

from app.dataset.frames import Frame, Frames


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

    def calculate_average_position(self, detections):
        if detections is None or len(detections) == 0:
            return None
        avg_x = 0
        count_x = 0
        avg_y = 0
        count_y = 0
        for det in detections:
            if det and len(det) > 0 and len(det[0]) > 1:
                avg_x += det[0][0]
                count_x += 1
                avg_y += det[0][1]
                count_y += 1

        if count_x > 0:
            avg_x /= count_x

        if count_y > 0:
            avg_y /= count_y

        return (avg_x, avg_y)

    def position_sorting_criteria(self, detection, avg_pos):
        if avg_pos is None:
            return 100 - detection[-1]
        # Calcola la distanza euclidea tra la detection e la media delle posizioni
        dist = np.sqrt(
            (detection[0] - avg_pos[0]) ** 2 + (detection[1] - avg_pos[1]) ** 2
        )
        return dist

    def _frame_detect(self, frame: Frame, avg_pos):
        gray_hist = cv2.equalizeHist(frame.gray)
        detections = self.classifier.detectMultiScale(gray_hist)

        detections = sorted(
            detections,
            key=lambda x: (self.position_sorting_criteria(x, avg_pos)),
            reverse=False,
        )

        # restituisce solo la migliore
        if detections:
            best_detection = detections[0]
            return [best_detection]
        else:
            print("No detections found")
            return []

    def detect(self):
        val = 1
        rects = []
        drawn_frames = []
        all_detections = []
        avg_pos = (0, 0)
        for frame in self.frames:
            val += 1
            avg_pos = self.calculate_average_position(all_detections)
            detections = self._frame_detect(frame, avg_pos)
            if detections == []:
                detections = all_detections[-1]
            all_detections.append(detections)
            rects.append(detections)
            frame = frame.rgb.copy()
            drawn_frames.append(frame)
            for i, rect in enumerate(detections):
                if i >= self.detections_to_keep:
                    break
                if rect is not None:
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Repeating the dectections for the first 10 frames with the new average position
        for i, frame in enumerate(self.frames[:10]):
            avg_pos = self.calculate_average_position(all_detections)
            detections = self._frame_detect(frame, avg_pos)
            if detections == []:
                detections = all_detections[-1]
            all_detections[i] = detections
            rects[i] = detections
            frame = frame.rgb.copy()
            drawn_frames[i] = frame
            for i, rect in enumerate(detections):
                if i >= self.detections_to_keep:
                    break
                if rect is not None:
                    x, y, w, h = rect
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return drawn_frames, rects

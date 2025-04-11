"""Module for Haar Cascade Object Detection."""

from typing import List

import cv2
import numpy as np

from handcrafted.app.dataset.frames import Frame, Frames


class HaarDetector:
    """Class for Haar Cascade Object Detection."""

    def __init__(
        self,
        frames: List["np.ndarray"],
        classifier,
        detections_to_keep: int = 1,
    ):
        """Initialize the HaarDetector object.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        classifier : cv2.CascadeClassifier
            The Haar Cascade classifier.
        detections_to_keep : int, optional
            The number of detections to keep, by default 1.

        """
        self.classifier = classifier
        self.frames = Frames(frames)
        self.detections_to_keep = detections_to_keep

    def calculate_average_position(self, detections):
        """Calculate the average position of the detected objects.

        This is used to make a more robust detection by using the average position, as the face tipically does not move a lot.

        Parameters
        ----------
        detections : List[List[int]]
            The list of detected objects.

        Returns
        -------
        Tuple[float, float]
            The average position of the detected objects.

        """
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
        """Sorting criteria for the detected objects.

        Parameters
        ----------
        detection : List[int]
            The detected object.
        avg_pos : Tuple[float, float]
            The average position of the detected objects.

        Returns
        -------
        float
            The distance between the detected object and the average position.

        """
        if avg_pos is None:
            return 100 - detection[-1]
        # Calculates the euclidean distance between the detection and the average position
        dist = np.sqrt(
            (detection[0] - avg_pos[0]) ** 2 + (detection[1] - avg_pos[1]) ** 2
        )
        return dist

    def _frame_detect(self, frame: Frame, avg_pos):
        """Detect objects in a single frame.

        Parameters
        ----------
        frame : Frame
            The frame to process.
        avg_pos : Tuple[float, float]
            The average position of the detected objects.

        Returns
        -------
        List[List[int]]
            The list of detected objects.

        """
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
            return []

    def detect(self):
        """Detect objects in all frames.

        Returns
        -------
        List[np.ndarray]
            The list of detected objects.

        """
        val = 1
        rects = []
        drawn_frames = []
        all_detections = []
        for frame in self.frames:
            val += 1
            avg_pos = self.calculate_average_position(all_detections)
            detections = self._frame_detect(frame, avg_pos)
            if detections == [] and all_detections != []:
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
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )
        # Repeating the dectections for the first 10 frames with the new average position
        for i, frame in enumerate(self.frames[:10]):  # type: ignore
            avg_pos = self.calculate_average_position(all_detections)
            detections = self._frame_detect(frame, avg_pos)
            if detections == [] and all_detections != []:
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
                    cv2.rectangle(
                        frame, (x, y), (x + w, y + h), (0, 255, 0), 2
                    )
        return drawn_frames, rects

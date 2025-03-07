from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np

from handcrafted.app.dataset.video import Video
from wlasl_mediapipe.app.utils.mp.helper.hand_landmark_drawer import (
    draw_landmarks_on_image,
)


class MediapipeLandmarksExtractor:
    """Wrapper for mediapipe Holistic class."""

    def __init__(
        self,
    ) -> None:
        """Initializes the MediapipeLandmarksExtractor object."""
        self.detector = mp.solutions.holistic.Holistic(  # type: ignore
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            refine_face_landmarks=True,
        )

    def _process_frame(self, frame: np.ndarray) -> List[np.ndarray]:
        """Process frame and return landmarks.

        Parameters
        ----------
        frame : np.ndarray
            The frame to process.

        Returns
        -------
        List[np.ndarray]
            The landmarks.
        """
        return self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def process_video(
        self, video: Video
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Process video and return landmarks.

        Parameters
        ----------
        video : Video
            The video to process.

        Returns
        -------
        Tuple[List[np.ndarray], List[np.ndarray]]
            The landmarks and the annotated frames.
        """
        landmarks = []
        annotated_frames = []
        for frame in video.get_frames():
            frame_landmarks = self._process_frame(frame)
            landmarks.append(frame_landmarks)
            annotated_frame = draw_landmarks_on_image(frame, frame_landmarks)
            annotated_frames.append(annotated_frame)
        self.detector.close()
        return landmarks, annotated_frames

from typing import List, Tuple
from handcrafted.app.dataset.video import Video
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2

from wlasl_mediapipe.app.utils.mp.helper.hand_landmark_drawer import (
    draw_hand_landmarks_on_image,
)


class MediapipeHandsHelper:
    """Wrapper for mediapipe Hands class"""

    def __init__(
        self,
        max_num_hands: int = 2,
    ) -> None:
        base_options = python.BaseOptions(
            model_asset_path="wlasl_mediapipe/mp_tasks/hand_landmarker.task"
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=max_num_hands
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _process_frame(self, frame: "np.ndarray") -> List["np.ndarray"] | None:
        return self.detector.detect(frame)

    def process_video(self, video: Video) -> Tuple[List, List["np.ndarray"]]:
        """Process video and return landmarks and annotated frames"""
        landmarks = []
        annotated_frames = []
        for frame in video.get_frames():
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            frame_landmarks = self._process_frame(frame_rgb)
            landmarks.append(frame_landmarks)
            print(frame_landmarks)
            annotated_frame = draw_hand_landmarks_on_image(frame, frame_landmarks)
            annotated_frames.append(annotated_frame)
        return landmarks, annotated_frames

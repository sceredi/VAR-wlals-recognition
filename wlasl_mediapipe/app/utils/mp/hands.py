from typing import List, Tuple
from handcrafted.app.dataset.video import Video
import mediapipe as mp

import numpy as np
import cv2


class MediapipeHandsHelper:
    """Wrapper for mediapipe Hands class"""
    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
    ) -> None:
        mp_hands = mp.solutions.hands
        self.hands = mp_hands(
            static_image_mode, max_num_hands, min_detection_confidence
        )

    def _process_frame(self, frame: "np.ndarray") -> List["np.ndarray"] | None:
        return self.hands.process(frame)

    def process_video(self, video: Video) -> Tuple[List, List["np.ndarray"]]:
        """Process video and return landmarks and drawn frames"""
        landmarks = []
        drawn_frames = []
        for frame in video.get_frames():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_landmarks = self._process_frame(frame_rgb)
            landmarks.append(frame_landmarks)
            drawn_frame = frame.copy()
            if frame_landmarks:
                for hand_landmarks in frame_landmarks.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        drawn_frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS
                    )
            drawn_frames.append(drawn_frame)
        return landmarks, drawn_frames

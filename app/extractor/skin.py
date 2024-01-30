# Purpose: Extracts skin from a video given a list of frames and the position of the face in each frame

from typing import List
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SkinExtractor:
    def __init__(self, frames: List["np.ndarray"], face_rects: List[List[int]]) -> None:
        self.frames = frames
        self.face_rects = face_rects

    def _extract_from_frame(self, frame: "np.ndarray", rect):
        if rect is None:
            return frame
        if len(rect) == 0:
            return frame
        # Extract head region from the frame
        x, y, w, h = rect[0]

        # Resticts the head region
        new_w = int(w * 0.6)
        new_h = int(h * 0.6)
        x = int(x + (w - new_w) / 2)
        y = int(y + (h - new_h) / 2)
        w = new_w
        h = new_h
        head_region = frame[y:y+h, x:x+w]
        
        # Convert head region to HSV color space
        hsv_head = cv2.cvtColor(head_region, cv2.COLOR_BGR2HSV)
        
        # Calculate mean color of the head region
        mean_color = np.mean(hsv_head, axis=(0, 1))
        print(f"Mean color: {mean_color}")
        
        # Define color range around the mean color
        tolerance = 50  # Adjust as needed
        lower_skin_hue = mean_color[0] - tolerance
        upper_skin_hue = mean_color[0] + tolerance
        lower_skin_saturation = mean_color[1] - tolerance
        upper_skin_saturation = mean_color[1] + tolerance
        lower_skin_value = mean_color[2] - tolerance
        upper_skin_value = mean_color[2] + tolerance
        if lower_skin_hue < 0:
            lower_skin_hue = 0
        if upper_skin_hue > 180:
            upper_skin_hue = 180
        if lower_skin_saturation < 0:
            lower_skin_saturation = 0
        if upper_skin_saturation > 255:
            upper_skin_saturation = 255
        if lower_skin_value < 0:
            lower_skin_value = 0
        if upper_skin_value > 255:
            upper_skin_value = 255
        lower_skin = np.array([lower_skin_hue, lower_skin_saturation, lower_skin_value], dtype=np.uint8)
        upper_skin = np.array([upper_skin_hue, upper_skin_saturation, upper_skin_value], dtype=np.uint8)
        print(f"Lower skin: {lower_skin}")
        print(f"Upper skin: {upper_skin}")
        
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Threshold the HSV image to get only skin color within the range
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply morphological closing to fill small holes in the mask
        kernel = np.ones((35, 35), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        new_frame = cv2.bitwise_and(frame, frame, mask=mask)
        

        
        
        return new_frame

    def extract(self):
        skin_frames = []
        for frame, rect in zip(self.frames, self.face_rects):
            skin_frame = self._extract_from_frame(frame, rect)
            skin_frames.append(skin_frame)
        return skin_frames

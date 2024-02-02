from typing import List
import cv2
import numpy as np


class ContourDetector:
    def __init__(self, frames: List["np.ndarray"]) -> None:
        self.frames = frames

    def detect(self) -> List["np.ndarray"]:
        result_frames = []
        for frame in self.frames:
            # Converti il frame in scala di grigi
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Applica un filtro di smoothing (ad esempio, filtro Gaussiano)
            blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)

            # Esegui la binarizzazione per enfatizzare i contorni delle mani
            _, binary_frame = cv2.threshold(blurred_frame, 100, 255, cv2.THRESH_BINARY)

            # Trova i contorni nell'immagine binarizzata
            contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Trova i contorni che potrebbero rappresentare le mani
            contours = [contour for contour in contours if cv2.contourArea(contour) > 1000]

            # Disegna i contorni sul frame originale
            result_frame = frame.copy()
            cv2.drawContours(result_frame, contours, -1, (0, 255, 0), 2)

            result_frames.append(result_frame)
        return result_frames

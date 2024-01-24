# Description: This file contains the class for calculating the optical flow

from typing import List
import cv2
import numpy as np

class FlowCalculator:
    def __init__(
        self,
        frames: List["np.ndarray"],
        feature_params=dict(
            maxCorners=50, qualityLevel=0.05, minDistance=5, blockSize=5
        ),
        lk_params=dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        ),
    ):
        self.frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        self.feature_params = feature_params
        self.lk_params = lk_params

    def _get_features(self, frame: "np.ndarray") -> "np.ndarray":
        return cv2.goodFeaturesToTrack(frame, mask=None, **self.feature_params)

    def calculate(self) -> List["np.ndarray"]:
        color = np.random.randint(0, 255, (self.feature_params['maxCorners'], 3))
        # Find corners in the first frame
        p0 = self._get_features(self.frames[0])
        # Create a mask image for drawing purposes
        mask = np.zeros_like(self.frames[0])
        flow = []
        for i in range(1, len(self.frames)):
            # Calculate optical flow between consecutive frames
            p1, st, _ = cv2.calcOpticalFlowPyrLK(
                self.frames[i-1], self.frames[i], p0, None, **self.lk_params
            )
            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]
            # Draw the tracks
            frame = self.frames[i].copy()
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)
            flow.append(img)
            # Update points for the next iteration
            p0 = good_new.reshape(-1, 1, 2)
        return flow


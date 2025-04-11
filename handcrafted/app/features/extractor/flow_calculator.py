"""Optical flow calculator using Farneback method."""

from typing import List

import cv2
import matplotlib.pyplot as plt
import numpy as np


class FlowCalculator:
    """Class to calculate optical flow using Farneback method."""

    def __init__(
        self,
        frames: List["np.ndarray"],
        last_frame_index: int,
        params=None,
    ):
        """Initialize the FlowCalculator object.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to process.
        last_frame_index : int
            The index of the last frame to process.
        params : dict, optional
            The parameters for the optical flow calculation, by default None.

        """
        if params is None:
            params = {
                "pyr_scale": 0.3,
                "levels": 5,
                "winsize": 10,
                "iterations": 6,
                "poly_n": 5,
                "poly_sigma": 1.5,
                "flags": 0,
            }
        self.frames = frames
        self.last_frame_index = last_frame_index
        self.params = params

    def calculate(self, plot_each_frame: bool = False):
        """Calculate the optical flow for the frames.

        Parameters
        ----------
        plot_each_frame : bool, optional
            Whether to plot each frame, by default False.

        Returns
        -------
        List[np.ndarray]
            The list of frames with optical flow applied.
        List[np.ndarray]
            The list of magnitudes of the optical flow.
        List[np.ndarray]
            The list of angles of the optical flow.

        """
        frames = []
        magnitudes = []
        angles = []
        prev_frame = cv2.cvtColor(self.frames[0], cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(self.frames[0])
        hsv[..., 1] = 255
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        frames.append(bgr)
        prev_flow = None
        for i, frame in enumerate(self.frames[1 : self.last_frame_index + 1]):
            next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_frame,
                next_frame,
                prev_flow,
                **self.params,  # type: ignore
            )
            prev_flow = flow
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            magnitudes.append(magnitude)
            angles.append(angle)
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = cv2.normalize(
                magnitude,
                None,
                0,
                255,
                cv2.NORM_MINMAX,  # type: ignore
            )
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            if plot_each_frame:
                self._plot(frame, bgr, i)
            frames.append(bgr)
            prev_frame = next_frame
        mask = np.zeros_like(self.frames[0])
        for _ in range(self.last_frame_index + 1, len(self.frames)):
            frames.append(mask)
        return frames, magnitudes, angles

    def _plot(self, frame, flow, index):
        """Plot the frame and flow.

        Parameters
        ----------
        frame : np.ndarray
            The frame to plot.
        flow : np.ndarray
            The flow to plot.
        index : int
            The index of the frame.

        """
        _fig, axes = plt.subplots(1, 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        axes[0].imshow(frame)
        axes[0].set_title(f"Frame {index}")

        axes[1].imshow(flow)
        axes[1].set_title(f"Flow {index}")
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        plt.show()

import os
from typing import List, Tuple

import cv2
import numpy as np
from scipy.interpolate import interp1d

from handcrafted.app.features.features_container import FeaturesContainer


class Video:
    """Class that represents a video in the dataset."""

    def __init__(
        self,
        gloss: str,
        video_id: str,
        split: str,
        signer_id: int,
        frame_start: int,
        frame_end: int,
        fps: int,
        bbox: List[int],
        sample: bool = False,
    ) -> None:
        """Initialize the video object.

        Parameters
        ----------
        gloss : str
            The gloss of the video.
        video_id : str
            The id of the video.
        split : str
            The split of the video, either "train", "val" or "test".
        signer_id : int
            The id of the signer.
        frame_start : int
            The frame of the video in which the sign starts.
        frame_end : int
            The frame of the video in which the sign ends.
        fps : int
            The frames per second of the video.
        bbox : List[int]
            The bounding box of the sign in the video, on the format [x_min, y_min, x_max, y_max].
        sample : bool, optional
            Whether the video is a sample video or not, by default False.
        """
        self.video_capture = None
        self.gloss = gloss
        self.video_id = video_id
        self.split = split
        self.signer_id = signer_id
        self.fps = fps
        self.bbox = bbox
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.features_container = FeaturesContainer(self, save=True)
        self._frames = None
        self._sample = sample

    @classmethod
    def from_instance(
        cls, gloss: str, instance: dict, is_sample: bool = False
    ) -> "Video":
        """Create a video object from an instance.

        Parameters
        ----------
        gloss : str
            The gloss of the video.
        instance : dict
            The instance from which to create the video object.
        is_sample : bool, optional
            Whether the video is a sample video or not, by default False.
        """
        return cls(
            gloss,
            instance["video_id"],
            instance["split"],
            instance["signer_id"],
            instance["frame_start"],
            instance["frame_end"],
            instance["fps"],
            instance["bbox"],
            sample=is_sample,
        )

    def __str__(self) -> str:
        """Return a string representation of the video object."""
        return f"image(gloss={self.gloss}, video_id={self.video_id}, split={self.split}, signer_id={self.signer_id}, frame_start={self.frame_start}, frame_end={self.frame_end}, fps={self.fps}, bbox={self.bbox})\n"

    def __repr__(self) -> str:
        """Return a string representation of the video object."""
        return str(self)

    def is_missing(self) -> bool:
        """Check if the video is missing from the available data."""
        return not os.path.isfile(self.get_path())

    def has_keypoints(self) -> bool:
        """Check if the video has keypoints already extracted with MediaPipe."""
        return os.path.exists(f"data/mp/{self.video_id}")

    def get_path(self) -> str:
        """Get the path to the video file."""
        if self._sample:
            return f"data/original_videos_sample/{self.video_id}.mp4"
        return f"data/videos/{self.video_id}.mp4"

    def get_video_capture(self) -> "cv2.VideoCapture":
        """Get the video capture object for the video."""
        if self.video_capture is None:
            self.video_capture = cv2.VideoCapture(self.get_path())
        return self.video_capture

    def get_frame(self, frame_number: int) -> Tuple[bool, "np.ndarray"]:
        """Get a frame from the video.

        Parameters
        ----------
        frame_number : int
            The frame number to get.

        Returns
        -------
        Tuple[bool, np.ndarray]
            A tuple with a boolean indicating if the frame was successfully read and the frame itself.
        """
        self.get_video_capture().set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.get_video_capture().read()
        return ret, frame

    def get_frames(self, last_frame=None) -> List["np.ndarray"]:
        """Get all frames from the video.

        Parameters
        ----------
        last_frame : int, optional
            The last frame to get, by default None, if None, all frames will be returned.

        Returns
        -------
        List[np.ndarray]
            A list with all frames from the video.
        """
        if self._frames is not None:
            return self._frames
        if last_frame is None:
            last_frame = self.frame_end
        frames = []
        frame_number = self.frame_start
        while frame_number < last_frame:
            ret, frame = self.get_frame(frame_number)
            if not ret:
                break
            frames.append(frame)
            frame_number += 1
        self.get_video_capture().release()
        self._frames = frames
        return frames

    def get_end(self) -> int:
        """Get the index of the last frame of the video.

        Returns
        -------
        int
            The index of the last frame of the video.
        """
        if self.frame_end != -1:
            return self.frame_end
        ret = self.get_video_capture().get(cv2.CAP_PROP_FRAME_COUNT)
        print(f"Video {self.video_id} has {ret} frames")
        self.get_video_capture().release()
        return int(ret)

    def __len__(self) -> int:
        """Return the number of frames in the video.

        Returns
        -------
        int
            The number of frames in the video.
        """
        return len(self.get_frames())

    def get_frames_padded(self, frames, target_num_frames=232) -> np.ndarray:
        """Pad the frames to a target number of frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to pad.
        target_num_frames : int, optional
            The target number of frames, by default 232.

        Returns
        -------
        np.ndarray
            The padded frames.
        """
        num_frames_to_add = target_num_frames - len(frames)
        last_frame = frames[-1]
        pad_frames = np.tile(last_frame, (num_frames_to_add, 1, 1, 1))
        new_frames = np.concatenate([frames, pad_frames])
        return new_frames

    def get_frames_interpolated(
        self, frames, target_num_frames=232
    ) -> List["np.ndarray"]:
        """Interpolate the frames to a target number of frames.

        Parameters
        ----------
        frames : List[np.ndarray]
            The frames to interpolate.
        target_num_frames : int, optional
            The target number of frames, by default 232.

        Returns
        -------
        List[np.ndarray]
            The interpolated frames.
        """
        x_old = np.linspace(0, 1, len(frames))
        x_new = np.linspace(0, 1, target_num_frames)

        # Calculates the interpolation function
        interpolator = interp1d(
            x_old, frames, kind="linear", axis=None, fill_value="extrapolate"  # type: ignore
        )

        # Applies the interpolation function to the new frames
        new_frames = interpolator(x_new)
        new_frames = (new_frames * 255.0).astype(np.uint8)

        return new_frames

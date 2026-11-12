"""RGB frame encoding without Torch, MoviePy or implicit image resizing."""

from collections.abc import Iterable, Sequence
from itertools import chain
from math import isfinite
from os import PathLike

import imageio.v2 as imageio
import numpy as np


def write_rgb_video(
    frames: Iterable[np.ndarray],
    path: str | PathLike[str],
    *,
    fps: float = 24,
    ffmpeg_params: Sequence[str] = (),
) -> None:
    """Encode uint8 RGB frames and close the FFmpeg writer on every exit."""
    if not isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    frames = iter(frames)
    first = next(frames, None)
    if first is None:
        raise ValueError("at least one video frame is required")
    if not isinstance(first, np.ndarray) or first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("frames must have shape (height, width, 3)")
    if first.dtype != np.uint8:
        raise ValueError("frames must use uint8 RGB values")
    height, width, _ = first.shape
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("yuv420p requires positive even height and width")
    with imageio.get_writer(
        str(path), format="FFMPEG", mode="I", fps=fps, codec="libx264",
        pixelformat="yuv420p", quality=None, macro_block_size=1,
        output_params=list(ffmpeg_params),
    ) as writer:
        for frame in chain((first,), frames):
            if not isinstance(frame, np.ndarray) or frame.shape != first.shape or frame.dtype != np.uint8:
                raise ValueError("all frames must have the same RGB shape and uint8 dtype")
            writer.append_data(frame)

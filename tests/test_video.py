import imageio.v2 as imageio
import numpy as np
import pytest

from depth_anything_3.utils import video


def test_real_ffmpeg_roundtrip_keeps_frame_count_and_dimensions(tmp_path):
    # Non-multiple of 16: the encoder must not silently resize to a macroblock.
    frames = [np.full((18, 22, 3), v, dtype=np.uint8) for v in (32, 220)]
    path = tmp_path / "roundtrip.mp4"
    video.write_rgb_video(frames, path, fps=12, ffmpeg_params=["-crf", "18"])
    with imageio.get_reader(path, format="FFMPEG") as reader:
        decoded = [frame for frame in reader]
        assert reader.get_meta_data()["fps"] == pytest.approx(12)
    assert len(decoded) == 2
    assert all(frame.shape == (18, 22, 3) for frame in decoded)
    assert float(decoded[0].mean()) < float(decoded[1].mean())


@pytest.mark.parametrize("frames", [[], [np.zeros((17, 22, 3), np.uint8)],
    [np.zeros((18, 22, 4), np.uint8)], [np.zeros((18, 22, 3), np.float32)]])
def test_bad_frames_rejected_before_opening_writer(monkeypatch, tmp_path, frames):
    monkeypatch.setattr(video.imageio, "get_writer", lambda *a, **k: pytest.fail("opened writer"))
    with pytest.raises(ValueError):
        video.write_rgb_video(frames, tmp_path / "bad.mp4")


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_bad_fps_rejected(tmp_path, fps):
    with pytest.raises(ValueError, match="fps"):
        video.write_rgb_video([], tmp_path / "bad.mp4", fps=fps)


def test_writer_closed_on_later_invalid_frame(monkeypatch, tmp_path):
    class Writer:
        closed = False
        count = 0
        def __enter__(self): return self
        def __exit__(self, *args): self.closed = True
        def append_data(self, frame): self.count += 1
    writer = Writer()
    monkeypatch.setattr(video.imageio, "get_writer", lambda *a, **kw: writer)
    frames = [np.zeros((18, 22, 3), np.uint8), np.zeros((20, 22, 3), np.uint8)]
    with pytest.raises(ValueError, match="same RGB shape"):
        video.write_rgb_video(frames, tmp_path / "bad.mp4")
    assert writer.closed and writer.count == 1

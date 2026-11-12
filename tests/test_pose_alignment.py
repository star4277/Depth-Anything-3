import numpy as np
import pytest

from depth_anything_3.utils.pose_align import _rand_rot, _umeyama_sim3_from_paths


def poses(centers):
    result = np.repeat(np.eye(4)[None], len(centers), axis=0)
    result[:, :3, 3] = centers
    return result


def test_recovers_known_scale_rotation_translation_and_pose_orientation():
    source = poses([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]])
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    scale, translation = 2.5, np.array([3, -2, 1])
    target = source.copy()
    target[:, :3, :3] = rotation
    target[:, :3, 3] = scale * (source[:, :3, 3] @ rotation.T) + translation
    r, t, s, aligned = _umeyama_sim3_from_paths(target, source)
    np.testing.assert_allclose(r, rotation, atol=1e-12)
    np.testing.assert_allclose(t, translation, atol=1e-12)
    assert s == pytest.approx(scale)
    np.testing.assert_allclose(aligned, target, atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1)


def test_planar_non_collinear_centers_are_supported():
    source = poses([[0, 0, 0], [1, 0, 0], [0, 2, 0]])
    r, t, s, aligned = _umeyama_sim3_from_paths(source, source)
    np.testing.assert_allclose(aligned, source, atol=1e-12)
    assert s == pytest.approx(1)


@pytest.mark.parametrize("centers", [[[0, 0, 0]] * 3, [[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 0, 0]]])
def test_degenerate_alignment_is_explicit(centers):
    with pytest.raises(ValueError):
        _umeyama_sim3_from_paths(poses(centers), poses(centers))


def test_random_rotation_works_with_numpy2():
    rotation = _rand_rot()
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1)

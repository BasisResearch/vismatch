"""Tests for vismatch model import, instantiation, and inference.

Covers the core matcher lifecycle exercised by vismatch_extract.py and
vismatch_match.py:

- Import: verify that the public API (get_matcher, available_models, BaseMatcher)
  is accessible.
- Instantiation: every model in ``available_models`` can be constructed on the
  current device; incompatible models are skipped gracefully.
- Inference (forward): matching two images via ``matcher.forward()`` returns the
  expected result dict with keypoints, descriptors, inlier counts, and optional
  per-match confidence.
- Inference (extract): single-image keypoint extraction via ``matcher.extract()``
  returns keypoints and descriptors.
"""

import importlib

import pytest
import numpy as np
import torch
from unittest.mock import patch

import vismatch
from vismatch import get_matcher, available_models, BaseMatcher


def test_import_vismatch_main():
    """Verify that the vismatch package exposes its public API."""
    assert hasattr(vismatch, "get_matcher")
    assert hasattr(vismatch, "available_models")
    assert hasattr(vismatch, "BaseMatcher")


def _mock_matcher(confidences):
    """Build a BaseMatcher whose _forward returns two matched kpts and the given confidence."""

    class Matcher(BaseMatcher):
        def _forward(self, img0, img1):
            kpts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
            empty = np.empty([0, 2])
            return kpts, kpts, empty, empty, empty, empty, confidences

    return Matcher()


@pytest.mark.parametrize(
    ("confidences", "expected_confidences"),
    [
        (None, None),
        (torch.tensor([0.25, 0.75]), np.array([0.25, 0.75])),
        (np.array([0.25, 0.75], dtype=np.float32), np.array([0.25, 0.75])),
    ],
)
def test_forward_matcher_confidences(test_images, confidences, expected_confidences):
    """forward() passes None confidence through and converts tensors/arrays to numpy."""
    result = _mock_matcher(confidences).forward(*test_images)
    if expected_confidences is None:
        assert result["matched_confidences"] is None
    else:
        assert isinstance(result["matched_confidences"], np.ndarray)
        np.testing.assert_allclose(result["matched_confidences"], expected_confidences)


def test_forward_requires_confidence_slot(test_images):
    """A matcher returning only 6 objects (missing the confidence slot) is rejected."""

    class Matcher(BaseMatcher):
        def _forward(self, img0, img1):
            return (None,) * 6  # missing the 7th object: matched_confidences

    with pytest.raises(AssertionError, match="must return 7 values"):
        Matcher().forward(*test_images)


def test_forward_bad_confidence_shape_fails(test_images):
    """Confidence not aligned 1:1 with the matched keypoints is rejected."""
    with pytest.raises(AssertionError):
        _mock_matcher(np.array([0.5], dtype=np.float32)).forward(*test_images)


def test_forward_removes_out_of_bounds_matches():
    """Matches with a keypoint outside either image (e.g. on regions added by padding) are removed,
    keeping confidences aligned (https://github.com/gmberton/vismatch/issues/69). img0 is 40x60
    (HxW) and img1 is 30x50, so only the first two (x, y) matches below are within bounds of both
    images (0 is inside, w/h and negatives are outside)."""
    kpts0 = np.array([[0.0, 0.0], [59.9, 39.9], [60.0, 10.0], [10.0, 45.0], [-0.5, 10.0], [10.0, 10.0]])
    kpts1 = np.array([[49.9, 29.9], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0], [10.0, 10.0], [50.0, 10.0]])
    confidences = np.linspace(0.1, 0.6, num=6)

    class Matcher(BaseMatcher):
        def _forward(self, img0, img1):
            empty = np.empty([0, 2])
            return kpts0, kpts1, empty, empty, empty, empty, confidences

    result = Matcher().forward(torch.rand(3, 40, 60), torch.rand(3, 30, 50))

    np.testing.assert_allclose(result["matched_kpts0"], kpts0[:2])
    np.testing.assert_allclose(result["matched_kpts1"], kpts1[:2])
    np.testing.assert_allclose(result["matched_confidences"], confidences[:2])


class _CornerMatcher(BaseMatcher):
    """Matches each image's bottom-right pixel to the other's, so every result identifies its pair by image size."""

    def _forward(self, img0, img1):
        (h0, w0), (h1, w1) = img0.shape[-2:], img1.shape[-2:]
        kpts0 = np.array([[w0 - 1, h0 - 1]], dtype=np.float32)
        kpts1 = np.array([[w1 - 1, h1 - 1]], dtype=np.float32)
        return kpts0, kpts1, kpts0, kpts1, None, None, None


@pytest.mark.parametrize(
    ("imgs0", "imgs1"),
    [
        ([torch.rand(3, 40, 60), torch.rand(3, 20, 30)], [torch.rand(3, 30, 50), torch.rand(3, 50, 70)]),
        (torch.rand(2, 3, 40, 60), torch.rand(2, 3, 30, 50)),
        (np.random.rand(2, 3, 40, 60).astype(np.float32), np.random.rand(2, 3, 30, 50).astype(np.float32)),
    ],
    ids=["list-mixed-sizes", "tensor", "numpy"],
)
def test_forward_batch_matches_each_pair(imgs0, imgs1):
    """A batch of pairs returns one result dict per pair, equal to matching that pair on its own."""
    matcher = _CornerMatcher()
    results = matcher.forward(imgs0, imgs1)

    assert isinstance(results, list) and len(results) == len(imgs0)
    for result, img0, img1 in zip(results, imgs0, imgs1):
        expected = matcher.forward(img0, img1)
        np.testing.assert_array_equal(result["matched_kpts0"], expected["matched_kpts0"])
        np.testing.assert_array_equal(result["matched_kpts1"], expected["matched_kpts1"])


@pytest.mark.parametrize(
    ("imgs0", "imgs1"),
    [
        (torch.rand(2, 3, 40, 60), torch.rand(3, 30, 50)),
        ([torch.rand(3, 40, 60)] * 3, [torch.rand(3, 30, 50)] * 2),
    ],
    ids=["batch-vs-single", "length-mismatch"],
)
def test_forward_batch_mismatch_fails(imgs0, imgs1):
    """img0 and img1 must both be single images or batches of the same length."""
    with pytest.raises(AssertionError, match="batches of the same length"):
        _CornerMatcher().forward(imgs0, imgs1)


def test_extract_batch():
    """extract() on a batch returns one result dict per image."""
    results = _CornerMatcher().extract([torch.rand(3, 40, 60), torch.rand(3, 20, 30)])
    assert [r["all_kpts0"].tolist() for r in results] == [[[59, 39]], [[29, 19]]]


def test_supports_batches_default():
    """Matchers default to looping over a batch one pair at a time."""
    assert _CornerMatcher().supports_batches is False


class _GridMatcher(BaseMatcher):
    """Detects a 3-point grid scaled to each image and matches keypoint i to keypoint i, natively batched."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_batches = True

    def _extract_features(self, imgs):
        feats = []
        for img in imgs:
            h, w = img.shape[-2:]
            kpts = torch.tensor([[0.0, 0.0], [w / 2, h / 2], [w - 1, h - 1]])
            feats.append({"all_kpts0": kpts, "all_desc0": torch.eye(3), "extra": kpts * 2})
        return feats

    def _match_features(self, feats0, feats1):
        idxs = torch.arange(3)
        return idxs, idxs, torch.ones(3)

    def _forward(self, img0, img1):
        f0, f1 = self._extract_features([img0, img1])
        return (
            f0["all_kpts0"],
            f1["all_kpts0"],
            f0["all_kpts0"],
            f1["all_kpts0"],
            f0["all_desc0"],
            f1["all_desc0"],
            None,
        )


def test_extract_native_matches_forward():
    """With supports_batches, extract() returns forward()'s keypoints and descriptors, the image size and model extras."""
    matcher, img = _GridMatcher(), torch.rand(3, 40, 60)
    feats, expected = matcher.extract(img), matcher.forward(img, img)
    np.testing.assert_array_equal(feats["all_kpts0"], expected["all_kpts0"])
    np.testing.assert_array_equal(feats["all_desc0"], expected["all_desc0"])
    assert feats["image_size"] == (60, 40)
    assert isinstance(feats["extra"], np.ndarray)


def test_extract_native_batch():
    """With supports_batches, extract() on a batch returns one result per image, each with its own image size."""
    results = _GridMatcher().extract([torch.rand(3, 40, 60), torch.rand(3, 20, 30)])
    assert [r["image_size"] for r in results] == [(60, 40), (30, 20)]
    assert [r["all_kpts0"][-1].tolist() for r in results] == [[59, 39], [29, 19]]


def test_match_matches_forward():
    """match() on extract() outputs gives forward()'s matches, plus the keypoint rows behind them."""
    matcher = _GridMatcher()
    matcher.skip_ransac = True
    img0, img1 = torch.rand(3, 40, 60), torch.rand(3, 30, 50)
    feats0, feats1 = matcher.extract(img0), matcher.extract(img1)
    result, expected = matcher.match(feats0, feats1), matcher.forward(img0, img1)

    np.testing.assert_array_equal(result["matched_kpts0"], expected["matched_kpts0"])
    np.testing.assert_array_equal(result["matched_kpts1"], expected["matched_kpts1"])
    np.testing.assert_array_equal(result["matched_kpts0"], feats0["all_kpts0"][result["matched_idxs0"]])
    np.testing.assert_array_equal(result["matched_kpts1"], feats1["all_kpts0"][result["matched_idxs1"]])
    assert result["num_inliers"] == expected["num_inliers"] == 0


def test_match_drops_out_of_bounds_matches():
    """match() drops matches with a keypoint outside its image, keeping indices aligned with keypoints."""
    matcher = _GridMatcher()
    feats0, feats1 = matcher.extract(torch.rand(3, 40, 60)), matcher.extract(torch.rand(3, 40, 60))
    feats1["image_size"] = (31, 21)  # only the (0, 0) and (30, 20) keypoints are inside
    result = matcher.match(feats0, feats1)
    assert result["matched_idxs1"].tolist() == [0, 1]
    assert result["matched_confidences"].tolist() == [1.0, 1.0]


def test_match_not_implemented():
    """match() needs a matcher with supports_batches; others must use forward()."""
    with pytest.raises(NotImplementedError, match="use forward"):
        _CornerMatcher().match({}, {})


@pytest.mark.parametrize("module, cls", [("xfeat", "xFeatMatcher"), ("loma", "LoMaMatcher")])
def test_match_features_not_sandboxed(module, cls):
    """_extract_features runs in the wrapper's ImportSandbox; per-pair _match_features skips its entry cost."""
    matcher_cls = getattr(importlib.import_module(f"vismatch.im_models.{module}"), cls)
    assert getattr(matcher_cls._extract_features, "_sandbox_key", None) is not None
    assert getattr(matcher_cls._match_features, "_sandbox_key", None) is None


@pytest.mark.parametrize("model_name", available_models)
def test_create_matcher(model_name, device):
    """Instantiate each available matcher and verify device assignment.

    Models that fail to instantiate (e.g. missing optional dependencies) are
    skipped rather than causing a test failure.
    """
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    assert matcher is not None
    assert matcher.device == device
    del matcher


@pytest.mark.parametrize("model_name", available_models)
def test_forward_synthetic_images(model_name, device, test_images):
    """Run matcher.forward() on a pair of synthetic random tensors.

    Validates that the result dict contains the required keys with correct
    dtypes and shapes: matched keypoints as (N, 2) numpy arrays, inlier count,
    and all-keypoints arrays.
    """
    img0, img1 = test_images
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")

    result = matcher.forward(img0, img1)

    assert isinstance(result, dict)
    assert "num_inliers" in result
    assert "matched_kpts0" in result
    assert "matched_kpts1" in result
    assert "matched_confidences" in result
    assert "all_kpts0" in result
    assert "all_kpts1" in result
    assert isinstance(result["matched_kpts0"], np.ndarray)
    assert isinstance(result["matched_kpts1"], np.ndarray)
    assert result["matched_confidences"] is None or isinstance(result["matched_confidences"], np.ndarray)
    assert result["matched_kpts0"].ndim == 2
    assert result["matched_kpts1"].ndim == 2
    if result["matched_confidences"] is not None:
        assert result["matched_confidences"].ndim == 1
        assert len(result["matched_confidences"]) == len(result["matched_kpts0"])
    if result["matched_kpts0"].shape[0] > 0:
        assert result["matched_kpts0"].shape[1] == 2
        assert result["matched_kpts1"].shape[1] == 2

    del matcher


@pytest.mark.parametrize("model_name", available_models)
def test_extract_keypoints(model_name, device, test_image_paths):
    """Run matcher.extract() on a single image to extract keypoints and descriptors.

    Validates the output dict contains ``all_kpts0`` as an (N, 2) numpy array
    and ``all_desc0`` for descriptors, matching the behaviour exercised by
    vismatch_extract.py.
    """
    img0_path, _ = test_image_paths
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")

    image = matcher.load_image(img0_path, resize=256)
    result = matcher.extract(image)

    assert isinstance(result, dict)
    assert "all_kpts0" in result
    assert "all_desc0" in result
    assert isinstance(result["all_kpts0"], np.ndarray)
    assert result["all_kpts0"].ndim == 2
    if result["all_kpts0"].shape[0] > 0:
        assert result["all_kpts0"].shape[1] == 2

    del matcher


def test_forward_batch_native():
    """With supports_batches, a batch of pairs matches through extract() and match(), equal to each pair on its own."""
    matcher = _GridMatcher()
    imgs0, imgs1 = [torch.rand(3, 40, 60), torch.rand(3, 20, 30)], [torch.rand(3, 30, 50), torch.rand(3, 50, 70)]
    with (
        patch.object(matcher, "_forward", wraps=matcher._forward) as forward_spy,
        patch.object(matcher, "_extract_features", wraps=matcher._extract_features) as extract_spy,
    ):
        results = matcher.forward(imgs0, imgs1)
    assert forward_spy.call_count == 0
    assert [len(call.args[0]) for call in extract_spy.call_args_list] == [2, 2]

    for result, img0, img1 in zip(results, imgs0, imgs1):
        expected = matcher.forward(img0, img1)
        for key in ("matched_kpts0", "matched_kpts1", "all_kpts0", "all_kpts1", "all_desc0", "all_desc1"):
            np.testing.assert_array_equal(result[key], expected[key])


def _overlap(kpts_a, kpts_b):
    """Fraction of keypoints in kpts_a that also appear in kpts_b (within 1e-3 px)."""
    if len(kpts_a) == 0:
        return 1.0 if len(kpts_b) == 0 else 0.0
    dists = np.linalg.norm(kpts_a[:, None] - kpts_b[None], axis=-1)
    return float((dists.min(1) < 1e-3).mean())


def _load_pair(matcher, test_image_paths):
    """The indoor test pair at 256 px, as (3, H, W) tensors."""
    return [matcher.load_image(p, resize=256) for p in test_image_paths]


@pytest.mark.parametrize("model_name", ["xfeat", "loma"])
def test_extract_matches_forward(model_name, device, test_image_paths):
    """For batching matchers, extract() returns exactly forward()'s keypoints and descriptors."""
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    assert matcher.supports_batches
    img, _ = _load_pair(matcher, test_image_paths)
    feats, expected = matcher.extract(img), matcher.forward(img, img)
    np.testing.assert_array_equal(feats["all_kpts0"], expected["all_kpts0"])
    np.testing.assert_array_equal(feats["all_desc0"], expected["all_desc0"])


@pytest.mark.parametrize("model_name", ["xfeat", "loma"])
def test_match_matches_forward_model(model_name, device, test_image_paths):
    """For batching matchers, match() on extract() outputs gives exactly forward()'s matches."""
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    matcher.skip_ransac = True
    img0, img1 = _load_pair(matcher, test_image_paths)
    feats0, feats1 = matcher.extract(img0), matcher.extract(img1)
    result, expected = matcher.match(feats0, feats1), matcher.forward(img0, img1)

    assert len(result["matched_kpts0"]) > 0
    np.testing.assert_array_equal(result["matched_kpts0"], expected["matched_kpts0"])
    np.testing.assert_array_equal(result["matched_kpts1"], expected["matched_kpts1"])
    if expected["matched_confidences"] is not None:
        np.testing.assert_array_equal(result["matched_confidences"], expected["matched_confidences"])
    np.testing.assert_array_equal(result["matched_kpts0"], feats0["all_kpts0"][result["matched_idxs0"]])
    np.testing.assert_array_equal(result["matched_kpts1"], feats1["all_kpts0"][result["matched_idxs1"]])


def test_extract_batch_xfeat(device, test_images):
    """xfeat extract() on a batch matches single-image extract(): exactly for one image, closely for several."""
    try:
        matcher = get_matcher("xfeat", device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate xfeat on {device}: {e}")
    singles = [matcher.extract(img) for img in test_images]

    (one,) = matcher.extract([test_images[0]])
    np.testing.assert_array_equal(one["all_kpts0"], singles[0]["all_kpts0"])
    np.testing.assert_array_equal(one["all_desc0"], singles[0]["all_desc0"])

    for batched, single in zip(matcher.extract(list(test_images)), singles):
        assert _overlap(batched["all_kpts0"], single["all_kpts0"]) >= 0.99


@pytest.mark.parametrize("model_name", ["xfeat", "loma"])
def test_forward_batch_native_model(model_name, device, test_image_paths):
    """A natively batched forward() matches nearly the same keypoints as forward() on each pair."""
    try:
        matcher = get_matcher(model_name, device=device)
    except Exception as e:
        pytest.skip(f"Cannot instantiate {model_name} on {device}: {e}")
    img0, img1 = _load_pair(matcher, test_image_paths)
    results = matcher.forward([img0, img1], [img1, img0])
    for result, (i0, i1) in zip(results, [(img0, img1), (img1, img0)]):
        expected = matcher.forward(i0, i1)
        assert _overlap(result["matched_kpts0"], expected["matched_kpts0"]) >= 0.99


def test_xfeat_supports_batches_sparse_only():
    """Only xfeat's sparse mode matches by descriptors alone, so only it batches natively."""
    try:
        flags = {name: get_matcher(name, device="cpu").supports_batches for name in ("xfeat", "xfeat-star")}
    except Exception as e:
        pytest.skip(f"Cannot instantiate xfeat: {e}")
    assert flags == {"xfeat": True, "xfeat-star": False}

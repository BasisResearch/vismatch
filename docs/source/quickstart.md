# Quickstart

## Matching

```python
from vismatch import get_matcher
from vismatch.viz import plot_matches

matcher = get_matcher("superpoint-lightglue", device="cuda")

img0 = matcher.load_image("img0.jpg", resize=512)
img1 = matcher.load_image("img1.jpg", resize=512)

result = matcher(img0, img1)
# result keys: num_inliers, H, all_kpts0, all_kpts1,
#              all_desc0, all_desc1, matched_kpts0, matched_kpts1,
#              inlier_kpts0, inlier_kpts1, matched_confidences
# matched_confidences is None if the matcher does not provide confidence.
# Confidence scores are matcher-specific and not normalized or comparable across matchers.

plot_matches(img0, img1, result, save_path="matches.png")
```

## Keypoint Extraction

```python
from vismatch.viz import plot_keypoints

result = matcher.extract(img0)
# result keys: all_kpts0, all_desc0

plot_keypoints(img0, result, save_path="kpts.png")
```

## Batch Matching

Pass a batch of pairs as `(B, 3, H, W)` tensors/arrays or as lists of images (paths, PIL Images, or
`(3, H, W)` tensors/arrays; sizes may differ within a list). The result is a list with one result dict per
pair; a single pair still returns a single dict.

```python
results = matcher([img0, img1], [img1, img0])
# results[0] matches img0 -> img1, results[1] matches img1 -> img0

kpts = matcher.extract([img0, img1])
# kpts[0]["all_kpts0"], kpts[1]["all_kpts0"]
```

## Ensemble Matching

Pass a list of matcher names to combine multiple models:

```python
matcher = get_matcher(["superpoint-lightglue", "disk-lightglue"], device="cuda")
result = matcher(img0, img1)
```

## Available Models

See {py:data}`vismatch.available_models` for the full list, or refer to the
[model details page](../source/model_details.md).

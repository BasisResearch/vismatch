import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from vismatch.import_sandbox import sandboxed_method
from vismatch.utils import to_normalized_coords, to_px_coords, to_numpy, _load_image, to_tensor_image, is_batch


class BaseMatcher(torch.nn.Module):
    """
    This serves as a base class for all matchers. It provides a simple interface
    for its sub-classes to implement, namely each matcher must specify its own
    __init__ and _forward methods. It also provides a common image_loader and
    homography estimator
    """

    def __init_subclass__(cls, **kwargs):
        # Run each wrapper-defined matcher's __init__, _forward and _extract_features inside that wrapper's
        # ImportSandbox: third-party code does lazy imports at construction time (MINIMA, EDM's
        # yacs configs) and at inference time (xfeat's lighterglue), and EnsembleMatcher /
        # Keypt2SubpxMatcher call inner matchers' _forward directly, bypassing forward().
        # _match_features stays outside: it runs once per pair on already built modules, and entering
        # the sandbox (tens of ms, it scans sys.modules) would cost far more than the match itself.
        super().__init_subclass__(**kwargs)
        if not cls.__module__.startswith("vismatch.im_models."):
            return
        for method_name in ("__init__", "_forward", "_extract_features"):
            method = cls.__dict__.get(method_name)
            if method is not None:
                setattr(cls, method_name, sandboxed_method(method, cls.__module__))

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__()
        self.device: str = device

        # Matchers that set this to True batch natively; others loop over a batch one pair at a time
        # A True matcher defines the two hooks behind extract() and match():
        #   _extract_features(imgs): (3, H, W) tensors on self.device -> one dict per image with all_kpts0 (N, 2),
        #     all_desc0 (N, D) and any tensor extras its _match_features needs
        #   _match_features(feats0, feats1): two such dicts on self.device -> (idxs0, idxs1, confidences or None),
        #     without modifying its inputs or importing; it runs outside the ImportSandbox
        self.supports_batches: bool = False

        self.skip_ransac: bool = False

        # OpenCV default ransac params
        self.ransac_iters: int = kwargs.get("ransac_iters", 2000)
        self.ransac_conf: float = kwargs.get("ransac_conf", 0.995)
        self.ransac_reproj_thresh: float = kwargs.get("ransac_reproj_thresh", 3)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def load_image(path: str | Path, resize: int | tuple = None, rot_angle: float = 0) -> torch.Tensor:
        """load image from filesystem and return as tensor. Optionally rotate and resize.

        Args:
            path (str | Path): path to image on filesystem
            resize (int | tuple, optional): size to resize img, either single value for square resize or tuple of (H, W). Defaults to None.
            rot_angle (float, optional): CCW rotation angle in degrees. Defaults to 0.

        Returns:
            torch.Tensor: image as tensor (C x H x W)
        """
        return _load_image(path=path, resize=resize, rot_angle=rot_angle)

    def rescale_coords(
        self,
        pts: np.ndarray | torch.Tensor,
        h_orig: int,
        w_orig: int,
        h_new: int,
        w_new: int,
    ) -> np.ndarray:
        """Rescale kpts coordinates from one img size to another

        Args:
            pts (np.ndarray | torch.Tensor): (N,2) array of kpts
            h_orig (int): height of original img
            w_orig (int): width of original img
            h_new (int): height of new img
            w_new (int): width of new img

        Returns:
            np.ndarray: (N,2) array of kpts in original img coordinates
        """
        return to_px_coords(to_normalized_coords(pts, h_new, w_new), h_orig, w_orig)

    def compute_ransac(
        self, matched_kpts0: np.ndarray, matched_kpts1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process matches into inliers and the respective Homography using RANSAC.

        Args:
            matched_kpts0 (np.ndarray): matching kpts from img0
            matched_kpts1 (np.ndarray): matching kpts from img1

        Returns:
            H (np.ndarray): (3 x 3) homography matrix from img0 to img1. Can be None if no homography is found
            inlier_kpts0 (np.ndarray): inlier kpts in img0
            inlier_kpts1 (np.ndarray): inlier kpts in img1
        """
        if len(matched_kpts0) < 4 or self.skip_ransac:  # Sperical matchers like sphereglue skip RANSAC
            return None, np.empty([0, 2]), np.empty([0, 2])

        H, inliers_mask = cv2.findHomography(
            matched_kpts0,
            matched_kpts1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self.ransac_reproj_thresh,
            maxIters=self.ransac_iters,
            confidence=self.ransac_conf,
        )
        inliers_mask = inliers_mask[:, 0].astype(bool)
        inlier_kpts0 = matched_kpts0[inliers_mask]
        inlier_kpts1 = matched_kpts1[inliers_mask]

        return H, inlier_kpts0, inlier_kpts1

    @torch.inference_mode()
    def forward(
        self,
        img0: torch.Tensor | np.ndarray | str | Path | Image.Image | list,
        img1: torch.Tensor | np.ndarray | str | Path | Image.Image | list,
    ) -> dict | list[dict]:
        """Run matching pipeline on two images, or on a batch of image pairs. All sub-classes implement this interface.

        Args:
            img0 (torch.Tensor | np.ndarray | str | Path | Image.Image | list): image as (3, H, W) array in [0, 1] range, path, or PIL Image;
                or a batch: a (B, 3, H, W) array or a list of such images
            img1 (torch.Tensor | np.ndarray | str | Path | Image.Image | list): same as img0; for a batch, pair i is (img0[i], img1[i])

        Returns:
            dict | list[dict]: result dict (for a batch, a list with one per pair) with keys:
                - num_inliers (int): number of inliers after RANSAC, i.e. len(inlier_kpts0)
                - H (np.ndarray): (3 x 3) homography matrix to map matched_kpts0 to matched_kpts1
                - all_kpts0 (np.ndarray): (N0 x 2) all detected keypoints from img0
                - all_kpts1 (np.ndarray): (N1 x 2) all detected keypoints from img1
                - all_desc0 (np.ndarray): (N0 x D) all descriptors from img0
                - all_desc1 (np.ndarray): (N1 x D) all descriptors from img1
                - matched_kpts0 (np.ndarray): (N2 x 2) keypoints from img0 that match matched_kpts1 (pre-RANSAC)
                - matched_kpts1 (np.ndarray): (N2 x 2) keypoints from img1 that match matched_kpts0 (pre-RANSAC)
                - inlier_kpts0 (np.ndarray): (N3 x 2) filtered matched_kpts0 that fit the H model (post-RANSAC)
                - inlier_kpts1 (np.ndarray): (N3 x 2) filtered matched_kpts1 that fit the H model (post-RANSAC)
                - matched_confidences (np.ndarray | None): (N2,) per-match confidence scores, None if the matcher does not provide confidence (pre-RANSAC).
        """

        # A batch of pairs is matched one pair at a time, unless the matcher batches natively
        if is_batch(img0) or is_batch(img1):
            assert is_batch(img0) and is_batch(img1) and len(img0) == len(img1), (
                "img0 and img1 must both be single images or batches of the same length"
            )
            if self.supports_batches:
                feats0, feats1 = self.extract(img0), self.extract(img1)
                return [
                    result
                    | {"all_kpts0": f0["all_kpts0"], "all_kpts1": f1["all_kpts0"]}
                    | {"all_desc0": f0["all_desc0"], "all_desc1": f1["all_desc0"]}
                    for result, f0, f1 in zip(self.match_batch(list(zip(feats0, feats1))), feats0, feats1)
                ]
            return [self.forward(i0, i1) for i0, i1 in zip(img0, img1)]

        # Take as input a pair of images
        img0 = to_tensor_image(img0).to(self.device)
        img1 = to_tensor_image(img1).to(self.device)

        # self._forward() is implemented by the children modules
        outputs = self._forward(img0, img1)
        assert len(outputs) == 7, (
            f"{self.name}._forward() must return 7 values "
            f"(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences), "
            f"got {len(outputs)}. Return None for matched_confidences if the matcher has no per-match confidence."
        )
        matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences = outputs

        # Check that returned objects are of accepted types (nd.array, torch.tensor or None)
        self.check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences)

        # Convert torch tensors to numpy. None objects stay None
        matched_kpts0, matched_kpts1 = to_numpy(matched_kpts0), to_numpy(matched_kpts1)
        all_kpts0, all_kpts1 = to_numpy(all_kpts0), to_numpy(all_kpts1)
        all_desc0, all_desc1 = to_numpy(all_desc0), to_numpy(all_desc1)
        matched_confidences = to_numpy(matched_confidences)

        # Some models might return kpts=None if no kpts are found. In this case, set an empty array with dim (0, 2)
        matched_kpts0 = self.get_empty_array_if_none(matched_kpts0)
        matched_kpts1 = self.get_empty_array_if_none(matched_kpts1)
        all_kpts0 = self.get_empty_array_if_none(all_kpts0)
        all_kpts1 = self.get_empty_array_if_none(all_kpts1)
        # Same for descriptors: if it is empty set as descriptor an array with dim (0, 2)
        all_desc0 = self.get_empty_array_if_none(all_desc0)
        all_desc1 = self.get_empty_array_if_none(all_desc1)

        # Check that shapes are correct and consistent
        self.check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences)

        # Drop matches with a kpt outside its image, e.g. on regions added by padding (see issue #69)
        (h0, w0), (h1, w1) = img0.shape[-2:], img1.shape[-2:]
        valid = (
            (matched_kpts0 >= 0) & (matched_kpts0 < [w0, h0]) & (matched_kpts1 >= 0) & (matched_kpts1 < [w1, h1])
        ).all(1)
        matched_kpts0, matched_kpts1 = matched_kpts0[valid], matched_kpts1[valid]
        if matched_confidences is not None:
            matched_confidences = matched_confidences[valid]

        # Compute RANSAC to obtain the inliers and homography matrix
        H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(matched_kpts0, matched_kpts1)

        return {
            "num_inliers": len(inlier_kpts0),
            "H": H,
            "all_kpts0": all_kpts0,
            "all_kpts1": all_kpts1,
            "all_desc0": all_desc0,
            "all_desc1": all_desc1,
            "matched_kpts0": matched_kpts0,
            "matched_kpts1": matched_kpts1,
            "inlier_kpts0": inlier_kpts0,
            "inlier_kpts1": inlier_kpts1,
            "matched_confidences": matched_confidences,
        }

    def extract(
        self, img: torch.Tensor | np.ndarray | str | Path | Image.Image | list
    ) -> dict[str, np.ndarray] | list[dict[str, np.ndarray]]:
        """Extract keypoints and descriptors from a single image, or from each image of a batch.

        Args:
            img (torch.Tensor | np.ndarray | str | Path | Image.Image | list): image as (3, H, W) array in [0, 1] range, path, or PIL Image;
                or a batch: a (B, 3, H, W) array or a list of such images

        Returns:
            dict | list[dict]: result dict (for a batch, a list with one per image) with keys:
                - all_kpts0 (np.ndarray): (N, 2) detected keypoints
                - all_desc0 (np.ndarray): (N, D) descriptors
                - image_size (tuple): (W, H) of the image, only for matchers with supports_batches
                - any model-specific extras that match() needs, only for matchers with supports_batches
        """
        # Matchers with native batching detect all images at once, skipping forward()'s self-pair match and RANSAC
        if self.supports_batches:
            imgs = [to_tensor_image(i).to(self.device) for i in (img if is_batch(img) else [img])]
            with torch.inference_mode():
                feats = self._extract_features(imgs)
            feats = [
                {**{k: to_numpy(v) for k, v in f.items()}, "image_size": (i.shape[-1], i.shape[-2])}
                for f, i in zip(feats, imgs)
            ]
            return feats if is_batch(img) else feats[0]

        if is_batch(img):
            return [self.extract(i) for i in img]
        result = self.forward(img, img)
        kpts = result["matched_kpts0"] if isinstance(self, EnsembleMatcher) else result["all_kpts0"]
        return {"all_kpts0": kpts, "all_desc0": result["all_desc0"]}

    def match(self, feats0: dict, feats1: dict) -> dict:
        """Match two images from their extract() outputs, without detecting again. Needs supports_batches.

        Args:
            feats0 (dict): extract() output for img0; values may be np.ndarray or torch.Tensor on any device
            feats1 (dict): extract() output for img1

        Returns:
            dict: result dict with the keys of forward() except all_kpts0/1 and all_desc0/1, plus:
                - matched_idxs0 (np.ndarray): (N2,) rows of feats0["all_kpts0"] behind matched_kpts0
                - matched_idxs1 (np.ndarray): (N2,) rows of feats1["all_kpts0"] behind matched_kpts1
        """
        return self.match_batch([(feats0, feats1)])[0]

    @torch.inference_mode()
    def match_batch(self, pairs: list[tuple[dict, dict]]) -> list[dict]:
        """Match many pairs of extract() outputs in one call, each result equal to match() on that pair.

        Matchers that override _match_features_batch() match all pairs in one forward; others loop
        _match_features(). Device-to-host copies happen once for the whole batch, so callers should
        pass as many pairs as fit in memory (xfeat: B x N0 x N1 floats, twice).

        Args:
            pairs (list[tuple[dict, dict]]): (feats0, feats1) extract() outputs; values may be np.ndarray or
                torch.Tensor on any device

        Returns:
            list[dict]: one match() result dict per pair, in order
        """
        if not self.supports_batches:
            raise NotImplementedError(f"{self.name} cannot match precomputed features, use forward()")
        if len(pairs) == 0:
            return []

        # Move features to the matcher's device, a no-op for features already there
        sizes = [(f0["image_size"], f1["image_size"]) for f0, f1 in pairs]
        feats0, feats1 = (
            [{k: torch.as_tensor(v, device=self.device) for k, v in f.items() if k != "image_size"} for f in side]
            for side in zip(*pairs)
        )

        # _match_features_batch() returns per pair indices into each keypoint table, and confidences or None
        matches = self._match_features_batch(feats0, feats1)

        # One device-to-host copy for every pair: concatenate, convert, split back by match count
        counts = [len(idxs0) for idxs0, _, _ in matches]
        splits = np.cumsum(counts)[:-1]
        has_conf = matches[0][2] is not None
        kpts0 = torch.cat([f["all_kpts0"][idxs0] for f, (idxs0, _, _) in zip(feats0, matches)])
        kpts1 = torch.cat([f["all_kpts0"][idxs1] for f, (_, idxs1, _) in zip(feats1, matches)])
        idxs0 = torch.cat([idxs0 for idxs0, _, _ in matches])
        idxs1 = torch.cat([idxs1 for _, idxs1, _ in matches])
        confs = torch.cat([conf for _, _, conf in matches]) if has_conf else None
        kpts0, kpts1, idxs0, idxs1 = (np.split(to_numpy(x), splits) for x in (kpts0, kpts1, idxs0, idxs1))
        confs = np.split(to_numpy(confs), splits) if has_conf else [None] * len(pairs)

        results = []
        for ((w0, h0), (w1, h1)), k0, k1, i0, i1, conf in zip(sizes, kpts0, kpts1, idxs0, idxs1, confs):
            # Drop matches with a kpt outside its image, as forward() does
            valid = ((k0 >= 0) & (k0 < [w0, h0]) & (k1 >= 0) & (k1 < [w1, h1])).all(1)
            k0, k1, i0, i1 = k0[valid], k1[valid], i0[valid], i1[valid]
            if conf is not None:
                conf = conf[valid]

            H, inlier_kpts0, inlier_kpts1 = self.compute_ransac(k0, k1)
            results.append(
                {
                    "num_inliers": len(inlier_kpts0),
                    "H": H,
                    "matched_kpts0": k0,
                    "matched_kpts1": k1,
                    "inlier_kpts0": inlier_kpts0,
                    "inlier_kpts1": inlier_kpts1,
                    "matched_confidences": conf,
                    "matched_idxs0": i0,
                    "matched_idxs1": i1,
                }
            )
        return results

    def _match_features_batch(self, feats0: list[dict], feats1: list[dict]) -> list[tuple]:
        """Match pair i of (feats0[i], feats1[i]) for every i; matchers that batch natively override this."""
        return [self._match_features(f0, f1) for f0, f1 in zip(feats0, feats1)]

    @staticmethod
    def get_empty_array_if_none(array: np.ndarray | None) -> np.ndarray:
        if array is None or array.size == 0:
            return np.empty([0, 2])
        return array

    @staticmethod
    def check_types(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences):
        """Check that objects are of accepted types (nd.array, torch.tensor or None)"""

        def is_array_or_tensor_or_none(data) -> bool:
            return data is None or isinstance(data, np.ndarray) or isinstance(data, torch.Tensor)

        assert is_array_or_tensor_or_none(matched_kpts0)
        assert is_array_or_tensor_or_none(matched_kpts1)
        assert is_array_or_tensor_or_none(all_kpts0)
        assert is_array_or_tensor_or_none(all_kpts1)
        assert is_array_or_tensor_or_none(all_desc0)
        assert is_array_or_tensor_or_none(all_desc1)
        assert is_array_or_tensor_or_none(matched_confidences)

    @staticmethod
    def check_shapes(matched_kpts0, matched_kpts1, all_kpts0, all_kpts1, all_desc0, all_desc1, matched_confidences):
        """Check that objects have appropriate shapes, e.g. keypoints should have shape (N, 2)"""

        def check_kpts_shape(np_array) -> bool:
            """Keypoint arrays should be in the form of N x 2"""
            return np_array.ndim == 2 and np_array.shape[1] == 2

        assert check_kpts_shape(matched_kpts0), f"matched_kpts0 shape should be (N x 2) but it is {matched_kpts0.shape}"
        assert check_kpts_shape(matched_kpts1), f"matched_kpts1 shape should be (N x 2) but it is {matched_kpts1.shape}"
        assert check_kpts_shape(all_kpts0), f"all_kpts0 shape should be (N x 2) but it is {all_kpts0.shape}"
        assert check_kpts_shape(all_kpts1), f"all_kpts1 shape should be (N x 2) but it is {all_kpts1.shape}"
        # Number of matched_kpts should be equal from both images
        assert matched_kpts0.shape == matched_kpts1.shape, f"{matched_kpts0.shape} != {matched_kpts1.shape}"
        # Descriptors should have shape (N x D)
        assert all_desc0.ndim == 2, str(all_desc0.shape)
        assert all_desc1.ndim == 2, str(all_desc1.shape)
        # Some models return no descriptors. If there are descriptors, there should be as many keypoints as descriptors.
        if all_desc0.shape[0] != 0:
            assert all_desc0.shape[0] == all_kpts0.shape[0], f"{all_desc0.shape[0]} != {all_kpts0.shape[0]}"
        if all_desc1.shape[0] != 0:
            assert all_desc1.shape[0] == all_kpts1.shape[0], f"{all_desc1.shape[0]} != {all_kpts1.shape[0]}"
        if matched_confidences is not None:
            assert matched_confidences.ndim == 1, (
                f"matched_confidences shape should be (N,) but it is {matched_confidences.shape}"
            )
            assert matched_confidences.shape[0] == matched_kpts0.shape[0], (
                f"{matched_confidences.shape[0]} != {matched_kpts0.shape[0]}"
            )


class EnsembleMatcher(BaseMatcher):
    def __init__(self, matcher_names: list[str] = [], device: str = "cpu", **kwargs):
        from vismatch import get_matcher

        super().__init__(device, **kwargs)
        self.matchers = [get_matcher(name, device=device, **kwargs) for name in matcher_names]

    def _forward(
        self, img0: torch.Tensor, img1: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, None, None, None, None, None]:
        all_matched_kpts0, all_matched_kpts1 = [], []
        for matcher in self.matchers:
            matched_kpts0, matched_kpts1, _, _, _, _, _ = matcher._forward(img0, img1)
            all_matched_kpts0.append(to_numpy(matched_kpts0))
            all_matched_kpts1.append(to_numpy(matched_kpts1))
        all_matched_kpts0, all_matched_kpts1 = np.concatenate(all_matched_kpts0), np.concatenate(all_matched_kpts1)
        return all_matched_kpts0, all_matched_kpts1, None, None, None, None, None

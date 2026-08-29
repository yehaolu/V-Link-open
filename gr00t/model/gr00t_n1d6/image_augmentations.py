from typing import Sequence
import warnings

import albumentations as A
import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms


def apply_with_replay(transform, images, masks=None, replay=None):
    """
    Apply albumentations transforms to multiple images with replay functionality.
    When masks are provided, mask-based transforms run per-frame before the main transform.

    Args:
        transform: Albumentations ReplayCompose or Compose transform
        images: List of PIL Images to transform
        masks: Optional list of masks aligned with images (H, W)
        replay: Optional replay data for consistent transforms. If None, creates new replay.

    Returns:
        tuple: (transformed_tensors_list, replay_data)
            - transformed_tensors_list: List of transformed torch tensors (C, H, W) as uint8
            - replay_data: Replay data for consistent transforms across images (None for regular Compose)
    """
    transformed_tensors = []
    current_replay = replay

    # Check if transform supports replay (ReplayCompose)
    has_replay = hasattr(transform, "replay")

    # Get mask-based transforms (applied per-frame, not replayed)
    mask_transforms = getattr(transform, "mask_transforms", None)

    if masks is not None and len(masks) != len(images):
        raise ValueError(
            f"Number of masks ({len(masks)}) must match number of images ({len(images)})"
        )

    for idx, img in enumerate(images):
        img_array = np.array(img)
        mask_array = None if masks is None else np.array(masks[idx])
        if mask_array is not None and mask_array.dtype == np.bool_:
            mask_array = mask_array.astype(np.uint8)

        # Apply mask-based transforms FIRST (per-frame, using current frame's mask)
        if mask_transforms and mask_array is not None:
            for mask_tf in mask_transforms:
                result = mask_tf(image=img_array, mask=mask_array)
                img_array = result["image"]

        if has_replay:
            if current_replay is None:
                # First image - create replay data
                augmented_image = transform(image=img_array)
                current_replay = augmented_image["replay"]
            else:
                # Subsequent images - use replay for consistent transforms
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning)
                    augmented_image = transform.replay(
                        image=img_array, saved_augmentations=current_replay
                    )
        else:
            # Regular Compose transform - no replay functionality
            augmented_image = transform(image=img_array)

        img_array = augmented_image["image"]
        # Convert to uint8 if needed (albumentations may return float32 in [0,1])
        if img_array.dtype == np.float32:
            img_array = (img_array * 255).astype(np.uint8)
        elif img_array.dtype != np.uint8:
            raise ValueError(f"Unexpected data type: {img_array.dtype}")

        # Convert to torch tensor (C, H, W) as uint8
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
        transformed_tensors.append(img_tensor)

    return transformed_tensors, current_replay


def _resize_depth_and_mask(
    depth: np.ndarray,
    valid_mask: np.ndarray,
    size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    target_h, target_w = size
    valid_float = valid_mask.astype(np.float32)
    weighted_depth = cv2.resize(
        depth.astype(np.float32) * valid_float,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    valid_weight = cv2.resize(
        valid_float,
        (target_w, target_h),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_depth = np.divide(
        weighted_depth,
        np.maximum(valid_weight, 1e-6),
        out=np.zeros_like(weighted_depth, dtype=np.float32),
        where=valid_weight > 1e-6,
    )
    return resized_depth.astype(np.float32), valid_weight > 0.5


def _smallest_max_size_shape(height: int, width: int, max_size) -> tuple[int, int]:
    if max_size is None:
        raise ValueError("SmallestMaxSize replay is missing max_size")
    if isinstance(max_size, (list, tuple)):
        max_size = max_size[0]
    scale = float(max_size) / float(min(height, width))
    return int(round(height * scale)), int(round(width * scale))


def _rotation_matrix(shape: tuple[int, int], angle: float) -> np.ndarray:
    height, width = shape
    center = (width / 2 - 0.5, height / 2 - 0.5)
    return cv2.getRotationMatrix2D(center, angle, 1.0)


def _crop_rotated_border(array: np.ndarray, transform_info: dict) -> np.ndarray:
    if not transform_info.get("crop_border", False):
        return array
    params = transform_info.get("params") or {}
    x_min = int(params["x_min"])
    x_max = int(params["x_max"])
    y_min = int(params["y_min"])
    y_max = int(params["y_max"])
    return array[y_min:y_max, x_min:x_max]


def apply_depth_with_replay(
    depth: np.ndarray,
    replay: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the geometric part of an albumentations replay to a depth map.

    RGB uses ReplayCompose for LetterBoxPad/resize/crop. Depth must follow the
    same geometry, but not photometric transforms such as ColorJitter. Depth is
    resized with validity-weighted bilinear interpolation so invalid pixels do
    not bleed into valid regions.
    """
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth map [height, width], got {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0.0).astype(np.float32)

    for transform_info in replay.get("transforms", []):
        if not transform_info.get("applied", False):
            continue
        name = transform_info.get("__class_fullname__", "").split(".")[-1]
        params = transform_info.get("params") or {}

        if name == "LetterBoxPad":
            height, width = depth.shape
            max_dim = max(height, width)
            default_top = (max_dim - height) // 2
            default_left = (max_dim - width) // 2
            pad_top = int(params.get("pad_top", default_top))
            pad_bottom = int(
                params.get("pad_bottom", max_dim - height - default_top)
            )
            pad_left = int(params.get("pad_left", default_left))
            pad_right = int(
                params.get("pad_right", max_dim - width - default_left)
            )
            depth = np.pad(
                depth,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=0.0,
            )
            valid = np.pad(
                valid,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=False,
            )
        elif name == "SmallestMaxSize":
            target_h, target_w = _smallest_max_size_shape(
                depth.shape[0],
                depth.shape[1],
                transform_info.get("max_size", None),
            )
            depth, valid = _resize_depth_and_mask(depth, valid, (target_h, target_w))
        elif name in {"FractionalRandomCrop", "FractionalCenterCrop"}:
            crop_coords = params.get("crop_coords")
            if crop_coords is None:
                raise ValueError(f"{name} replay is missing crop_coords")
            x_min, y_min, x_max, y_max = [int(v) for v in crop_coords]
            depth = depth[y_min:y_max, x_min:x_max]
            valid = valid[y_min:y_max, x_min:x_max]
        elif name == "Rotate":
            matrix = _rotation_matrix(depth.shape, float(params["angle"]))
            border_mode = int(
                transform_info.get("border_mode", cv2.BORDER_REFLECT_101)
            )
            height, width = depth.shape
            weighted_depth = cv2.warpAffine(
                depth * valid.astype(np.float32),
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
                borderValue=0.0,
            )
            valid_weight = cv2.warpAffine(
                valid.astype(np.float32),
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=border_mode,
                borderValue=0.0,
            )
            depth = np.divide(
                weighted_depth,
                np.maximum(valid_weight, 1e-6),
                out=np.zeros_like(weighted_depth, dtype=np.float32),
                where=valid_weight > 1e-6,
            )
            valid = valid_weight > 0.5
            depth = _crop_rotated_border(depth, transform_info)
            valid = _crop_rotated_border(valid, transform_info)
        else:
            # Photometric transforms such as ColorJitter intentionally do not
            # apply to metric depth.
            continue

    return depth.astype(np.float32), valid.astype(np.float32)


def apply_segmentation_with_replay(
    segmentation: np.ndarray,
    replay: dict,
    ignore_index: int = 255,
) -> np.ndarray:
    """
    Apply the geometric part of an albumentations replay to a label map.

    This must follow RGB/depth geometry exactly, but label maps use nearest
    interpolation and padded pixels are marked as ignore_index.
    """
    segmentation = np.asarray(segmentation)
    if segmentation.ndim != 2:
        raise ValueError(f"Expected segmentation map [height, width], got {segmentation.shape}")
    output_dtype = segmentation.dtype

    for transform_info in replay.get("transforms", []):
        if not transform_info.get("applied", False):
            continue
        name = transform_info.get("__class_fullname__", "").split(".")[-1]
        params = transform_info.get("params") or {}

        if name == "LetterBoxPad":
            height, width = segmentation.shape
            max_dim = max(height, width)
            default_top = (max_dim - height) // 2
            default_left = (max_dim - width) // 2
            pad_top = int(params.get("pad_top", default_top))
            pad_bottom = int(
                params.get("pad_bottom", max_dim - height - default_top)
            )
            pad_left = int(params.get("pad_left", default_left))
            pad_right = int(
                params.get("pad_right", max_dim - width - default_left)
            )
            segmentation = np.pad(
                segmentation,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode="constant",
                constant_values=int(ignore_index),
            )
        elif name == "SmallestMaxSize":
            target_h, target_w = _smallest_max_size_shape(
                segmentation.shape[0],
                segmentation.shape[1],
                transform_info.get("max_size", None),
            )
            segmentation = cv2.resize(
                segmentation,
                (target_w, target_h),
                interpolation=cv2.INTER_NEAREST,
            )
        elif name in {"FractionalRandomCrop", "FractionalCenterCrop"}:
            crop_coords = params.get("crop_coords")
            if crop_coords is None:
                raise ValueError(f"{name} replay is missing crop_coords")
            x_min, y_min, x_max, y_max = [int(v) for v in crop_coords]
            segmentation = segmentation[y_min:y_max, x_min:x_max]
        elif name == "Rotate":
            matrix = _rotation_matrix(
                segmentation.shape,
                float(params["angle"]),
            )
            height, width = segmentation.shape
            segmentation = cv2.warpAffine(
                segmentation,
                matrix,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=int(
                    transform_info.get("border_mode", cv2.BORDER_REFLECT_101)
                ),
                borderValue=int(ignore_index),
            )
            segmentation = _crop_rotated_border(
                segmentation,
                transform_info,
            )
        else:
            continue

    return segmentation.astype(output_dtype, copy=False)


class MaskedColorTransform(A.ImageOnlyTransform):
    """Apply random tint to specific mask regions.

    Args:
        target_mask_values: List of mask values to apply the transform to
        alpha_range: (min, max) for random_tint overlay intensity
        p: Probability of applying the transform
    """

    def __init__(
        self,
        target_mask_values: Sequence[int],
        alpha_range: tuple[float, float] = (0.3, 1.0),
        p: float = 0.5,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.target_mask_values = list(target_mask_values)
        self.alpha_range = alpha_range

    def apply(self, img: np.ndarray, mask: np.ndarray = None, **params) -> np.ndarray:
        if mask is None:
            return img

        region_mask = np.zeros(mask.shape[:2], dtype=bool)
        for val in self.target_mask_values:
            region_mask |= mask == val

        if not region_mask.any():
            return img

        # Random color
        random_color = np.random.randint(0, 256, size=3).astype(np.float32)
        result = img.copy().astype(np.float32)

        # Random tint: semi-transparent overlay
        alpha = np.random.uniform(self.alpha_range[0], self.alpha_range[1])
        for c in range(3):
            result[region_mask, c] = result[region_mask, c] * (1 - alpha) + random_color[c] * alpha

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_params_dependent_on_data(self, params, data) -> dict:
        return {"mask": data.get("mask")}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("target_mask_values", "alpha_range")


class BackgroundNoiseTransform(A.ImageOnlyTransform):
    """Replace specified mask regions with random noise.

    This transform replaces pixels where mask value matches target_mask_values with random RGB noise,
    useful for domain randomization in sim-to-real transfer.

    Args:
        p: Probability of applying the transform
        target_mask_values: Mask values to replace with noise (default: [0])
    """

    def __init__(
        self,
        p: float = 1.0,
        target_mask_values: Sequence[int] | None = None,
        always_apply: bool | None = None,
    ):
        super().__init__(p=p, always_apply=always_apply)
        self.target_mask_values = [0] if target_mask_values is None else list(target_mask_values)

    def apply(self, img: np.ndarray, mask: np.ndarray = None, **params) -> np.ndarray:
        if mask is None:
            return img

        result = img.copy()
        mask_2d = mask[..., 0] if mask.ndim == 3 else mask
        background = np.isin(mask_2d, self.target_mask_values)

        if background.any():
            noise = np.random.randint(0, 256, size=result.shape, dtype=np.uint8)
            result[background] = noise[background]

        return result

    def get_params_dependent_on_data(self, params, data) -> dict:
        return {"mask": data.get("mask")}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("target_mask_values",)


class FractionalRandomCrop(A.DualTransform):
    """Crop a random part of the input based on fractions while maintaining aspect ratio.

    Args:
        crop_fraction: Fraction of the image to crop (0.0 to 1.0). The crop will maintain
                      the original aspect ratio and be this fraction of the original area.
        p: probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes, crop_coords, params["shape"]
        )

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        # Calculate crop dimensions with linear scaling
        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)

        # Ensure minimum size of 1x1
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)
        # Random position for crop
        max_y = height - crop_height
        max_x = width - crop_width

        y_min = np.random.randint(0, max_y + 1) if max_y > 0 else 0
        x_min = np.random.randint(0, max_x + 1) if max_x > 0 else 0

        crop_coords = (x_min, y_min, x_min + crop_width, y_min + crop_height)
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


class FractionalCenterCrop(A.DualTransform):
    """Crop the center part of the input based on fractions while maintaining aspect ratio.

    Args:
        crop_fraction: Fraction of the image to crop (0.0 to 1.0). The crop will maintain
                      the original aspect ratio and be this fraction of the original area.
        p: probability of applying the transform. Default: 1.0

    Targets:
        image, mask, bboxes, keypoints

    Image types:
        uint8, float32
    """

    def __init__(
        self, crop_fraction: float = 0.9, p: float = 1.0, always_apply: bool | None = None
    ):
        super().__init__(p=p, always_apply=always_apply)
        if not 0.0 < crop_fraction <= 1.0:
            raise ValueError("crop_fraction must be between 0.0 and 1.0")
        self.crop_fraction = crop_fraction

    def apply(
        self, img: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        x_min, y_min, x_max, y_max = crop_coords
        return img[y_min:y_max, x_min:x_max]

    def apply_to_bboxes(
        self, bboxes: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_bboxes_by_coords(
            bboxes, crop_coords, params["shape"]
        )

    def apply_to_keypoints(
        self, keypoints: np.ndarray, crop_coords: tuple[int, int, int, int], **params
    ) -> np.ndarray:
        return A.augmentations.crops.functional.crop_keypoints_by_coords(keypoints, crop_coords)

    def get_params_dependent_on_data(self, params, data) -> dict[str, tuple[int, int, int, int]]:
        image_shape = params["shape"][:2]
        height, width = image_shape

        # Calculate crop dimensions with linear scaling
        crop_height = int(height * self.crop_fraction)
        crop_width = int(width * self.crop_fraction)

        # Ensure minimum size of 1x1
        crop_height = max(1, crop_height)
        crop_width = max(1, crop_width)

        # Center the crop
        y_min = (height - crop_height) // 2
        x_min = (width - crop_width) // 2

        crop_coords = (x_min, y_min, x_min + crop_width, y_min + crop_height)
        return {"crop_coords": crop_coords}

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ("crop_fraction",)


class LetterBoxPad(A.DualTransform):
    """Pad non-square images to square by adding black bars (letterboxing).

    This is the albumentations equivalent of LetterBoxTransform (torchvision).
    Ensures all images have the same spatial dimensions after padding,
    regardless of their original aspect ratio.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, p: float = 1.0, always_apply: bool | None = None):
        super().__init__(p=p, always_apply=always_apply)

    def apply(
        self,
        img: np.ndarray,
        pad_top: int = 0,
        pad_bottom: int = 0,
        pad_left: int = 0,
        pad_right: int = 0,
        **params,
    ) -> np.ndarray:
        if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
            return img
        return cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0
        )

    def get_params_dependent_on_data(self, params, data) -> dict[str, int]:
        h, w = params["shape"][:2]
        if h == w:
            return {"pad_top": 0, "pad_bottom": 0, "pad_left": 0, "pad_right": 0}
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w
        return {
            "pad_top": pad_h // 2,
            "pad_bottom": pad_h - pad_h // 2,
            "pad_left": pad_w // 2,
            "pad_right": pad_w - pad_w // 2,
        }

    def get_transform_init_args_names(self) -> tuple[str, ...]:
        return ()


def build_image_transformations_albumentations(
    image_target_size,
    image_crop_size,
    random_rotation_angle,
    color_jitter_params,
    shortest_image_edge,
    crop_fraction,
    extra_augmentation_config: dict | None = None,
):
    """
    Build albumentations-based image transformations equivalent to the torchvision version.

    Args:
        image_target_size: Target size for resizing (list of [height, width])
        image_crop_size: Size for cropping (list of [height, width])
        random_rotation_angle: Maximum rotation angle in degrees (0 for no rotation)
        color_jitter_params: Dictionary with color jitter parameters (brightness, contrast, saturation, hue)
        shortest_image_edge: Shortest edge size for resizing
        crop_fraction: Fraction of image to crop
        extra_augmentation_config: Optional dict for additional augmentations. Supported keys:
            - "background_noise_transforms": list of dicts, each with:
                - "target_mask_values": list of int (e.g., [0])
                - "p": float (probability of applying transform)
            - "masked_region_transforms": list of dicts, each with:
                - "target_mask_values": list of int (e.g., [4] or [5])
                - "p": float (probability of applying transform)
                - "alpha_range": [min, max] for random_tint mode intensity

    Returns:
        tuple: (train_transform, eval_transform) - raw albumentations transforms
    """

    if crop_fraction is None:
        fraction_to_use = image_crop_size[0] / image_target_size[0]
    else:
        fraction_to_use = crop_fraction

    if shortest_image_edge is None:
        max_size = image_target_size[0]
    else:
        max_size = shortest_image_edge

    extra_augmentation_config = extra_augmentation_config or {}

    # Training transforms (using ReplayCompose for consistent augmentation across views)
    # Use SmallestMaxSize to preserve aspect ratios, with INTER_AREA for antialiasing
    train_transform_list = [
        LetterBoxPad(),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        FractionalRandomCrop(crop_fraction=fraction_to_use),
        A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
    ]

    if random_rotation_angle is not None and random_rotation_angle != 0:
        train_transform_list.append(A.Rotate(limit=random_rotation_angle, p=1.0))

    if color_jitter_params is not None:
        # Map torchvision ColorJitter parameters to albumentations ColorJitter
        # Note: albumentations uses different parameter names and ranges
        train_transform_list.append(
            A.ColorJitter(
                brightness=color_jitter_params.get("brightness", 0.0),
                contrast=color_jitter_params.get("contrast", 0.0),
                saturation=color_jitter_params.get("saturation", 0.0),
                hue=color_jitter_params.get("hue", 0.0),
                p=1.0,
            )
        )

    train_transform = A.ReplayCompose(train_transform_list, p=1.0)

    # === Mask-based augmentations (applied per-frame, NOT in ReplayCompose) ===
    # These transforms depend on per-frame mask data and must not be replayed
    # to ensure each frame uses its own mask
    mask_transforms = []

    # Background noise on mask regions
    for noise_cfg in extra_augmentation_config.get("background_noise_transforms", []):
        target_mask_values = noise_cfg.get("target_mask_values", [0])
        p = noise_cfg.get("p", 1.0)
        mask_transforms.append(
            BackgroundNoiseTransform(
                p=float(p),
                target_mask_values=target_mask_values,
            )
        )

    # Masked region transforms
    for transform_cfg in extra_augmentation_config.get("masked_region_transforms", []):
        target_mask_values = transform_cfg.get("target_mask_values", [])
        p = transform_cfg.get("p", 0.5)
        alpha_range = tuple(transform_cfg.get("alpha_range", [0.3, 1.0]))

        mask_transforms.append(
            MaskedColorTransform(
                target_mask_values=target_mask_values,
                alpha_range=alpha_range,
                p=p,
            )
        )

    # Attach mask transforms to the main transform for use in apply_with_replay
    train_transform.mask_transforms = mask_transforms if mask_transforms else None

    # Evaluation transforms (deterministic, no extra augmentations)
    # Use SmallestMaxSize to preserve aspect ratios, with INTER_AREA for antialiasing
    eval_transform = A.ReplayCompose(
        [
            LetterBoxPad(),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
            FractionalCenterCrop(crop_fraction=fraction_to_use),
            A.SmallestMaxSize(max_size=max_size, interpolation=cv2.INTER_AREA),
        ]
    )

    return train_transform, eval_transform


class LetterBoxTransform:
    """Custom transform to pad non-square images to square by adding black bars.

    Works with any tensor shape where the last 3 dimensions are (C, H, W).
    Leading dimensions (batch, time, views, etc.) are preserved.
    """

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Pad image to square dimensions by adding black bars to the smaller dimension.

        Args:
            img: Image tensor of shape (..., C, H, W) where ... can be any leading dimensions
                 Examples: (C, H, W), (B, C, H, W), (B, T*V, C, H, W)

        Returns:
            Padded image tensor of shape (..., C, max(H,W), max(H,W))
        """
        # Get the height and width from the last 2 dimensions
        *leading_dims, c, h, w = img.shape

        if h == w:
            return img

        # Calculate padding needed
        max_dim = max(h, w)
        pad_h = max_dim - h
        pad_w = max_dim - w

        # Add padding to center the image (divide padding equally on both sides)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        # If we have leading dimensions, we need to flatten them, pad, then unflatten
        if leading_dims:
            # Reshape to (batch, C, H, W) where batch includes all leading dimensions
            batch_size = torch.tensor(leading_dims).prod().item()
            img_reshaped = img.reshape(batch_size, c, h, w)

            # Apply padding to each image in the batch
            # torchvision padding format: (left, right, top, bottom)
            padded_img = transforms.functional.pad(
                img_reshaped, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )

            # Reshape back to original leading dimensions
            output_shape = leading_dims + [c, max_dim, max_dim]
            padded_img = padded_img.reshape(output_shape)
        else:
            # Simple case: just (C, H, W)
            padded_img = transforms.functional.pad(
                img, padding=[pad_left, pad_top, pad_right, pad_bottom], fill=0
            )

        return padded_img


def build_image_transformations(
    image_target_size, image_crop_size, random_rotation_angle, color_jitter_params
):
    """
    Build torchvision-based image transformations.

    Args:
        image_target_size: Target size for resizing (list of [height, width])
        image_crop_size: Size for cropping (list of [height, width])
        random_rotation_angle: Maximum rotation angle in degrees (0 for no rotation)
        color_jitter_params: Dictionary with color jitter parameters (brightness, contrast, saturation, hue)

    Returns:
        tuple: (train_transform, eval_transform) - torchvision transforms
    """
    transform_list = [
        transforms.ToImage(),
        LetterBoxTransform(),
        # transforms.ToDtype(torch.get_default_dtype(), scale=True),
        transforms.Resize(size=image_target_size),
        transforms.RandomCrop(size=image_crop_size),
        transforms.Resize(size=image_target_size),
    ]
    if random_rotation_angle is not None and random_rotation_angle != 0:
        transform_list.append(
            transforms.RandomRotation(degrees=[-random_rotation_angle, random_rotation_angle])
        )
    if color_jitter_params is not None:
        transform_list.append(transforms.ColorJitter(**color_jitter_params))
    train_image_transform = transforms.Compose(transform_list)
    eval_image_transform = transforms.Compose(
        [
            transforms.ToImage(),
            # transforms.ToDtype(torch.get_default_dtype(), scale=True),
            LetterBoxTransform(),
            transforms.Resize(size=image_target_size),
            transforms.CenterCrop(size=image_crop_size),
            transforms.Resize(size=image_target_size),
        ]
    )
    return train_image_transform, eval_image_transform

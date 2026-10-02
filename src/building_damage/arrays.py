"""Read numeric final-tile arrays safely and validate each batch."""

from pathlib import Path
import numpy as np


def open_arrays(images_path, masks_path, image_size=512):
    images = np.load(images_path, mmap_mode="r", allow_pickle=False)
    masks = np.load(masks_path, mmap_mode="r", allow_pickle=False)
    if images.dtype != np.uint8 or masks.dtype != np.uint8:
        raise ValueError("Expected uint8 RGB images and integer class masks; object arrays are unsupported.")
    if images.ndim != 4 or images.shape[1:] != (image_size, image_size, 3):
        raise ValueError("Image array must have shape (N, image_size, image_size, 3).")
    if masks.shape != images.shape[:3] or len(images) == 0:
        raise ValueError("Mask array must match (N, image_size, image_size), with at least one sample.")
    return images, masks


def batch(images, masks, indexes, classes=5):
    labels = np.asarray(masks[indexes])
    if labels.min() < 0 or labels.max() >= classes:
        raise ValueError("Mask IDs must be in the configured class range; labels are never clipped.")
    x = np.asarray(images[indexes], dtype=np.float32) / np.float32(255.0)
    y = np.eye(classes, dtype=np.float32)[labels]
    return x, y


def metadata(images_path, masks_path, images, masks):
    return {
        "images": {"path": str(Path(images_path).resolve()), "shape": list(images.shape), "dtype": str(images.dtype)},
        "masks": {"path": str(Path(masks_path).resolve()), "shape": list(masks.shape), "dtype": str(masks.dtype)},
        "header_validation_only": True,
        "historical_split_provenance_verified": False,
    }

"""Explicit, streamed xBD post-disaster scenes for new local experiments.

This is a later cleanup workflow, not the missing historical preparation driver.
Manifest creation reads filenames and file metadata only. Scene loading retains
all four 512-pixel tiles, performs no resizing or augmentation, and never loads
an entire split into memory.
"""

from hashlib import sha256
import io
import json
from pathlib import Path
import re

import cv2
import numpy as np
import shapely.wkt
import tifffile


PROTOCOL = "fresh_postdisaster_all_four_tiles_noaug"
SPLITS = ("tier1", "tier3", "hold", "test")
UNKNOWN_POLICIES = ("error", "ignore", "undamaged_legacy")
CLASS_MAP = {"0": "background", "1": "undamaged", "2": "minor-damage",
             "3": "major-damage", "4": "destroyed"}
SUBTYPE_CLASSES = {"no-damage": 1, "minor-damage": 2,
                   "major-damage": 3, "destroyed": 4}
_SCENE_ID = re.compile(r"[A-Za-z0-9_-]+_post_disaster\Z")


def _policy(value):
    if not isinstance(value, str) or value not in UNKNOWN_POLICIES:
        raise ValueError(f"unknown_policy must be one of {UNKNOWN_POLICIES}")
    return value


def _scene_id(value):
    if not isinstance(value, str) or not _SCENE_ID.fullmatch(value):
        raise ValueError("scene_id must be a post-disaster filename stem")
    return value


def _unique_sequence(values, name, allowed=None):
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a sequence, not one string")
    values = list(values)
    if not values or any(not isinstance(value, str) for value in values):
        raise ValueError(f"{name} must contain nonempty string values")
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be nonempty with no duplicates")
    if allowed is not None and any(value not in allowed for value in values):
        raise ValueError(f"{name} must contain only {allowed}")
    return values


def _stat(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"missing scene input: {path}")
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _check_stat(path, expected):
    if (not isinstance(expected, dict) or set(expected) != {"size", "mtime_ns"}
            or any(type(expected[key]) is not int or expected[key] < 0
                   for key in expected)):
        raise ValueError("invalid manifest input stat record")
    if _stat(path) != expected:
        raise ValueError(f"scene input changed since manifest creation: {path}")


def create_manifest(root, splits, unknown_policy="error", scene_ids=None):
    """Return a deterministic v1 manifest without reading image/label payloads.

    ``root`` is the directory containing tier1/tier3/hold/test, usually geotiffs.
    Optional scene IDs are full stems including ``_post_disaster``. Selection
    preserves complete scenes; no arbitrary first-N or pre-disaster selection
    is performed. Every post image/label pair in the selected splits is checked
    for matching names, including when an explicit subset is requested.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset root is not a directory: {root}")
    splits = sorted(_unique_sequence(splits, "splits", SPLITS))
    unknown_policy = _policy(unknown_policy)
    selected = None
    if scene_ids is not None:
        selected = set(_scene_id(value) for value in
                       _unique_sequence(scene_ids, "scene_ids"))
    entries = []
    seen = set()
    for split in splits:
        image_dir, label_dir = root / split / "images", root / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"split requires images/ and labels/ directories: {split}")
        images = {path.stem: path for path in image_dir.glob("*_post_disaster.tif")}
        labels = {path.stem: path for path in label_dir.glob("*_post_disaster.json")}
        missing_images, missing_labels = labels.keys() - images.keys(), images.keys() - labels.keys()
        if missing_images or missing_labels:
            raise ValueError(f"unmatched post-disaster inputs in {split}: "
                             f"{len(missing_images)} missing images, "
                             f"{len(missing_labels)} missing labels")
        for scene_id in sorted(images):
            _scene_id(scene_id)
            if scene_id in seen:
                raise ValueError(f"scene_id occurs in multiple selected splits: {scene_id}")
            seen.add(scene_id)
            if selected is not None and scene_id not in selected:
                continue
            image, labels_path = images[scene_id].resolve(), labels[scene_id].resolve()
            entries.append({"split": split, "scene_id": scene_id,
                            "image": str(image), "labels": str(labels_path),
                            "image_stat": _stat(image), "labels_stat": _stat(labels_path)})
    if selected is not None and selected - seen:
        raise ValueError(f"requested scene IDs not found: {sorted(selected - seen)}")
    if not entries:
        raise ValueError("selected dataset contains no paired post-disaster scenes")
    return {"version": 1, "protocol": PROTOCOL, "dataset_root": str(root),
            "class_map": dict(CLASS_MAP), "splits": splits,
            "unknown_policy": unknown_policy, "scenes": entries}


def _validate_entry(entry):
    if not isinstance(entry, dict):
        raise ValueError("manifest scene entry must be an object")
    required = {"split", "scene_id", "image", "labels", "image_stat", "labels_stat"}
    if not required.issubset(entry):
        raise ValueError("manifest scene entry is missing required fields")
    if entry["split"] not in SPLITS:
        raise ValueError("invalid manifest split")
    _scene_id(entry["scene_id"])
    for key, extension in (("image", ".tif"), ("labels", ".json")):
        value = entry[key]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError("manifest scene paths must be absolute")
        if Path(value).name != entry["scene_id"] + extension:
            raise ValueError("manifest input filename does not match scene_id")
        _check_stat(value, entry[key + "_stat"])


def load_manifest(path):
    """Read a manifest and validate its protocol, pairs, paths and input stats.

    Input contents are not hashed until actual scene loading. A size/mtime
    check catches normal changes but is not a cryptographic identity check.
    """
    with Path(path).open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if (not isinstance(manifest, dict) or type(manifest.get("version")) is not int
            or manifest["version"] != 1 or manifest.get("protocol") != PROTOCOL
            or manifest.get("class_map") != CLASS_MAP):
        raise ValueError("unsupported or invalid scene manifest schema/protocol")
    _policy(manifest.get("unknown_policy"))
    splits = _unique_sequence(manifest.get("splits", []), "splits", SPLITS)
    root_value = manifest.get("dataset_root")
    if not isinstance(root_value, str) or not Path(root_value).is_absolute():
        raise ValueError("manifest dataset_root must be absolute")
    root = Path(root_value)
    scenes = manifest.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("manifest must contain nonempty scenes")
    seen = set()
    for entry in scenes:
        _validate_entry(entry)
        if entry["split"] not in splits or entry["scene_id"] in seen:
            raise ValueError("manifest has unlisted split or duplicate scene_id")
        seen.add(entry["scene_id"])
        for key, directory, extension in (("image", "images", ".tif"),
                                          ("labels", "labels", ".json")):
            expected = (root / entry["split"] / directory /
                        (entry["scene_id"] + extension)).resolve()
            if Path(entry[key]).resolve() != expected:
                raise ValueError("manifest input path does not match dataset root/split")
    if scenes != sorted(scenes, key=lambda item: (item["split"], item["scene_id"])):
        raise ValueError("manifest scenes must be sorted by split and scene_id")
    return manifest


def _contour(ring):
    coordinates = np.asarray(ring.coords, dtype=np.float64)
    if (coordinates.ndim != 2 or coordinates.shape[1] != 2
            or not np.isfinite(coordinates).all()
            or np.abs(coordinates).max(initial=0) >= np.iinfo(np.int32).max):
        raise ValueError("annotation coordinates must be finite 2D pixel positions")
    # Historical helper used astype(int32): truncate toward zero, without x2.
    return coordinates.astype(np.int32)


def _rasterize(annotation, unknown_policy):
    if not isinstance(annotation, dict) or not isinstance(annotation.get("features"), dict):
        raise ValueError("annotation must contain features.xy")
    features = annotation["features"].get("xy")
    if not isinstance(features, list):
        raise ValueError("annotation features.xy must be a list")
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    subtype_counts, unknown_counts = {}, {}
    for index, feature in enumerate(features):
        if not isinstance(feature, dict) or not isinstance(feature.get("properties"), dict):
            raise ValueError(f"invalid annotation properties at index {index}")
        properties = feature["properties"]
        if properties.get("feature_type", "building") != "building":
            raise ValueError(f"non-building xy annotation at index {index}")
        subtype = properties.get("subtype")
        if subtype is not None and not isinstance(subtype, str):
            raise ValueError(f"damage subtype must be a string at index {index}")
        name = "<missing>" if subtype is None else str(subtype)
        subtype_counts[name] = subtype_counts.get(name, 0) + 1
        if subtype not in SUBTYPE_CLASSES:
            unknown_counts[name] = unknown_counts.get(name, 0) + 1
            if unknown_policy == "error":
                raise ValueError(f"unknown damage subtype {name!r} at index {index}; "
                                 "choose an explicit unknown_policy")
            label = 255 if unknown_policy == "ignore" else 1
        else:
            label = SUBTYPE_CLASSES[subtype]
        try:
            geometry = shapely.wkt.loads(feature["wkt"])
        except (KeyError, TypeError, ValueError, shapely.errors.GEOSException) as exc:
            raise ValueError(f"invalid annotation WKT at index {index}") from exc
        if (geometry.geom_type not in ("Polygon", "MultiPolygon")
                or geometry.is_empty or not geometry.is_valid or geometry.has_z):
            raise ValueError(f"invalid 2D polygon geometry at index {index}")
        polygons = geometry.geoms if geometry.geom_type == "MultiPolygon" else [geometry]
        for polygon in polygons:
            contours = [_contour(polygon.exterior)]
            contours.extend(_contour(ring) for ring in polygon.interiors)
            # OpenCV even-odd filling preserves holes; overlapping annotations
            # are processed in JSON order and later annotations win.
            cv2.fillPoly(mask, contours, color=int(label))
    return mask, len(features), subtype_counts, unknown_counts


def load_scene(entry, unknown_policy="error"):
    """Load one complete scene: uint8 RGB, class mask, and content provenance.

    Integer TIFFs in [0,255] (including the local int16 TIFFs) are accepted.
    Floating point, out-of-range, wrong-sized or non-RGB inputs fail explicitly;
    no implicit wrapping, scaling, resizing or band reordering is performed.
    Ignore policy marks unknown building pixels as 255. A consumer must exclude
    these pixels from metrics and loss rather than treating them as background.
    """
    unknown_policy = _policy(unknown_policy)
    _validate_entry(entry)
    image_bytes = Path(entry["image"]).read_bytes()
    label_bytes = Path(entry["labels"]).read_bytes()
    image = tifffile.imread(io.BytesIO(image_bytes))
    if image.shape != (1024, 1024, 3) or not np.issubdtype(image.dtype, np.integer):
        raise ValueError("scene image must be integer RGB with shape (1024,1024,3)")
    if image.min() < 0 or image.max() > 255:
        raise ValueError("scene RGB values must lie in [0,255]; refuse uint8 wrapping")
    source_dtype = str(image.dtype)
    image = image.astype(np.uint8)
    annotation = json.loads(label_bytes)
    if not isinstance(annotation, dict):
        raise ValueError("annotation JSON root must be an object")
    metadata = annotation.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("annotation metadata must be an object")
    for key in ("width", "height", "original_width", "original_height"):
        if key in metadata and metadata[key] != 1024:
            raise ValueError("annotation metadata dimensions do not match scene image")
    mask, count, subtype_counts, unknown_counts = _rasterize(annotation, unknown_policy)
    # Detect ordinary concurrent mutations while the payload was read.
    _check_stat(entry["image"], entry["image_stat"])
    _check_stat(entry["labels"], entry["labels_stat"])
    provenance = {"image_sha256": sha256(image_bytes).hexdigest(),
                  "labels_sha256": sha256(label_bytes).hexdigest(),
                  "image_shape": list(image.shape), "source_image_dtype": source_dtype,
                  "returned_image_dtype": "uint8", "mask_dtype": "uint8",
                  "annotation_count": count, "subtype_counts": subtype_counts,
                  "unknown_counts": unknown_counts, "unknown_policy": unknown_policy,
                  "rasterization": "OpenCV fillPoly; xy truncated toward zero; "
                                   "holes preserved; later JSON annotations win"}
    return image, mask, provenance


def iter_tiles(entry, image, mask):
    """Yield four paired tiles in TL, TR, BL, BR order, keeping empty tiles.

    ``tile_id`` is 0..3; ``row``/``col`` are original-scene pixel offsets.
    Tiles are views of the loaded scene, not persisted array copies.
    """
    if image.shape != (1024, 1024, 3) or mask.shape != (1024, 1024):
        raise ValueError("four-tile workflow requires 1024x1024 RGB and mask")
    for tile_id, (row, col) in enumerate(((0, 0), (0, 512), (512, 0), (512, 512))):
        yield {"tile_id": tile_id, "image": image[row:row + 512, col:col + 512],
               "mask": mask[row:row + 512, col:col + 512], "row": row, "col": col,
               "split": entry["split"], "scene_id": entry["scene_id"]}

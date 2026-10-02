"""Synthetic raw-scene checks; no real xBD evaluation or model training."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import tifffile

from building_damage.scenes import create_manifest, iter_tiles, load_manifest, load_scene


def feature(subtype, wkt):
    properties = {"feature_type": "building"}
    if subtype is not None:
        properties["subtype"] = subtype
    return {"properties": properties, "wkt": wkt}


def square(x, y, width=20):
    return (f"POLYGON (({x} {y}, {x + width} {y}, {x + width} {y + width}, "
            f"{x} {y + width}, {x} {y}))")


class SceneContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "geotiffs"
        self.scene_id = "synthetic-event_00000001_post_disaster"
        self.annotations = [feature(name, square(10 + 30 * i, 10)) for i, name in enumerate(
            ("no-damage", "minor-damage", "major-damage", "destroyed"))]
        self.write_scene(self.scene_id, annotations=self.annotations)

    def tearDown(self):
        self.temp.cleanup()

    def write_scene(self, scene_id, split="test", image=None, annotations=None):
        folder = self.root / split
        (folder / "images").mkdir(parents=True, exist_ok=True)
        (folder / "labels").mkdir(exist_ok=True)
        if image is None:
            image = np.full((1024, 1024, 3), 200, dtype=np.int16)
        tifffile.imwrite(folder / "images" / f"{scene_id}.tif", image, photometric="rgb")
        annotation = {"features": {"xy": self.annotations if annotations is None else annotations},
                      "metadata": {"width": 1024, "height": 1024}}
        (folder / "labels" / f"{scene_id}.json").write_text(json.dumps(annotation))

    def entry(self, policy="error"):
        return create_manifest(self.root, ["test"], policy)["scenes"][0]

    def test_manifest_uses_metadata_only_and_post_selection_is_explicit(self):
        self.write_scene("synthetic-event_00000001_pre_disaster", annotations=[])
        self.write_scene("synthetic-event_00000002_post_disaster", annotations=[])
        with patch("building_damage.scenes.tifffile.imread", side_effect=AssertionError("payload read")):
            manifest = create_manifest(self.root, ["test"], scene_ids=[self.scene_id])
        self.assertEqual([item["scene_id"] for item in manifest["scenes"]], [self.scene_id])
        self.assertEqual(manifest["version"], 1)
        path = Path(self.temp.name) / "manifest.json"
        path.write_text(json.dumps(manifest))
        self.assertEqual(load_manifest(path), manifest)
        with self.assertRaisesRegex(ValueError, "post-disaster"):
            create_manifest(self.root, ["test"], scene_ids=["synthetic-event_00000001_pre_disaster"])

    def test_missing_pairs_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            create_manifest(self.root, ["test", "test"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            create_manifest(self.root, ["test"], scene_ids=[self.scene_id, self.scene_id])
        self.write_scene(self.scene_id, split="hold")
        with self.assertRaisesRegex(ValueError, "multiple"):
            create_manifest(self.root, ["hold", "test"])
        (self.root / "test" / "labels" / f"{self.scene_id}.json").unlink()
        with self.assertRaisesRegex(ValueError, "unmatched"):
            create_manifest(self.root, ["test"])

    def test_stale_or_tampered_manifest_fails_before_reading_payload(self):
        manifest = create_manifest(self.root, ["test"])
        path = Path(self.temp.name) / "manifest.json"
        path.write_text(json.dumps(manifest))
        labels = Path(manifest["scenes"][0]["labels"])
        labels.write_text(labels.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "changed"):
            load_manifest(path)
        manifest = create_manifest(self.root, ["test"])
        manifest["class_map"]["1"] = "destroyed"
        path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "schema"):
            load_manifest(path)

    def test_int16_rgb_and_all_classes_have_content_provenance(self):
        rgb, mask, provenance = load_scene(self.entry())
        self.assertEqual(rgb.dtype, np.uint8)
        self.assertTrue(np.all(rgb == 200))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertEqual(set(np.unique(mask)), {0, 1, 2, 3, 4})
        for i in range(4):
            self.assertEqual(mask[20, 20 + 30 * i], i + 1)
        self.assertEqual(provenance["source_image_dtype"], "int16")
        self.assertEqual(len(provenance["image_sha256"]), 64)
        self.assertEqual(len(provenance["labels_sha256"]), 64)

    def test_unknown_and_missing_subtypes_require_explicit_policy(self):
        self.write_scene(self.scene_id, annotations=[feature("un-classified", square(10, 10)),
                                                       feature(None, square(50, 10))])
        with self.assertRaisesRegex(ValueError, "unknown damage"):
            load_scene(self.entry())
        _, ignored, provenance = load_scene(self.entry("ignore"), "ignore")
        self.assertEqual(ignored[20, 20], 255)
        self.assertEqual(ignored[20, 60], 255)
        self.assertEqual(provenance["unknown_counts"], {"un-classified": 1, "<missing>": 1})
        _, legacy, _ = load_scene(self.entry("undamaged_legacy"), "undamaged_legacy")
        self.assertEqual(legacy[20, 20], 1)
        self.assertEqual(legacy[20, 60], 1)
        with self.assertRaisesRegex(ValueError, "unknown_policy"):
            create_manifest(self.root, ["test"], "silently-background")

    def test_holes_truncation_and_later_annotation_precedence(self):
        polygon = ("POLYGON ((10.9 10.9, 40.9 10.9, 40.9 40.9, 10.9 40.9, 10.9 10.9), "
                   "(20 20, 20 30, 30 30, 30 20, 20 20))")
        self.write_scene(self.scene_id, annotations=[feature("minor-damage", polygon),
                                                    feature("destroyed", square(35, 35, 10))])
        _, mask, _ = load_scene(self.entry())
        self.assertEqual(mask[10, 10], 2)  # Truncation, rather than rounding to 11.
        self.assertEqual(mask[25, 25], 0)  # Interior hole stays background.
        self.assertEqual(mask[37, 37], 4)  # Later JSON polygon wins in overlap.
        self.assertEqual(mask[15, 15], 2)

    def test_invalid_geometry_and_unsafe_rgb_are_rejected(self):
        self.write_scene(self.scene_id, annotations=[feature("no-damage", "POLYGON ((0 0, 20 20, 0 20, 20 0, 0 0))")])
        with self.assertRaisesRegex(ValueError, "geometry"):
            load_scene(self.entry())
        self.write_scene(self.scene_id, image=np.full((1024, 1024, 3), 300, dtype=np.int16))
        with self.assertRaisesRegex(ValueError, "wrapping"):
            load_scene(self.entry())
        self.write_scene(self.scene_id, image=np.zeros((1024, 1024, 3), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "integer RGB"):
            load_scene(self.entry())
        self.write_scene(self.scene_id, image=np.zeros((512, 512, 3), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "shape"):
            load_scene(self.entry())

    def test_four_tiles_keep_pairing_and_empty_tiles(self):
        rgb = np.zeros((1024, 1024, 3), dtype=np.uint8)
        mask = np.zeros((1024, 1024), dtype=np.uint8)
        offsets = ((0, 0), (0, 512), (512, 0), (512, 512))
        for i, (row, col) in enumerate(offsets):
            rgb[row:row + 512, col:col + 512] = 20 * (i + 1)
            mask[row:row + 512, col:col + 512] = i
        tiles = list(iter_tiles(self.entry(), rgb, mask))
        self.assertEqual(len(tiles), 4)
        for i, tile in enumerate(tiles):
            self.assertEqual((tile["row"], tile["col"]), offsets[i])
            self.assertEqual(tile["tile_id"], i)
            self.assertTrue(np.all(tile["image"] == 20 * (i + 1)))
            self.assertTrue(np.all(tile["mask"] == i))
            self.assertTrue(np.shares_memory(tile["image"], rgb))
        self.assertTrue(np.all(tiles[0]["mask"] == 0))


if __name__ == "__main__":
    unittest.main()

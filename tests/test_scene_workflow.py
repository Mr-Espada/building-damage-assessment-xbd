"""Meaningful run-contract checks without training or recovered-data evaluation."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from building_damage.__main__ import main
from building_damage.scene_workflow import targets, validate_training_manifests, evaluate


class SceneWorkflow(unittest.TestCase):
    def test_ignored_targets_and_invalid_ids(self):
        labels = np.array([[[0, 4, 255]]], dtype=np.uint8)
        y = targets(labels)
        np.testing.assert_array_equal(y.sum(-1), [[[1, 1, 0]]])
        self.assertEqual(y[0, 0, 1, 4], 1)
        with self.assertRaises(ValueError):
            targets(np.array([[[6]]], dtype=np.uint8))

    def test_scene_overlap_and_policy_mismatch(self):
        first = {"scenes": [{"scene_id": "scene_post_disaster", "image": "/train.tif"}], "unknown_policy": "ignore"}
        second = {"scenes": [{"scene_id": "scene_post_disaster", "image": "/val.tif"}], "unknown_policy": "ignore"}
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_training_manifests(first, second)
        second["scenes"][0]["scene_id"] = "other_post_disaster"
        second["unknown_policy"] = "error"
        with self.assertRaisesRegex(ValueError, "same unknown"):
            validate_training_manifests(first, second)

    def test_scene_plan_has_no_execution_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            cfg = Path(__file__).resolve().parents[1] / "configs/local_inference.json"
            prefix = root / "best_model.ckpt"
            Path(str(prefix) + ".index").touch()
            Path(str(prefix) + ".data-00000-of-00001").touch()
            value = {"scenes": [{"scene_id": "a_post_disaster"}], "unknown_policy": "ignore", "protocol": "new"}
            args = ["evaluate-scenes", "--manifest", str(manifest), "--config", str(cfg),
                    "--checkpoint", str(prefix), "--output-dir", str(root / "run")]
            with patch("building_damage.scene_workflow.load_manifest", return_value=value), contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0)
            self.assertFalse((root / "run").exists())
            plan = json.JSONDecoder().raw_decode(output.getvalue())[0]
            self.assertEqual(plan["tile_count"], 4)
            self.assertEqual(plan["status"], "new_run_plan_not_historical_reproduction")

    def test_evaluation_masks_unknowns_and_keeps_score_families_distinct(self):
        # Deterministic fake inference isolates the runner's masking/aggregation contract.
        from types import SimpleNamespace
        masks = [np.array([[0, 255]], dtype=np.uint8), np.array([[1, 4]], dtype=np.uint8)]
        batches = [[{"image": np.zeros((1, 2, 3), dtype=np.uint8), "mask": m}] for m in masks]
        probs = iter([np.eye(5, dtype=np.float32)[np.array([[[0, 3]]])],
                      np.eye(5, dtype=np.float32)[np.array([[[1, 4]]])]])
        def checked(y, p):
            np.testing.assert_array_equal(y.sum(-1), p.sum(-1))
            return np.float32(.5)
        sm = SimpleNamespace(losses=SimpleNamespace(DiceLoss=lambda: checked),
                             metrics=SimpleNamespace(iou_score=checked, f1_score=checked))
        with tempfile.TemporaryDirectory() as temp, patch("building_damage.scene_workflow.scene_batches", return_value=iter(batches)):
            measured = evaluate(lambda x, training=False: next(probs), sm,
                                {"scenes": [1], "unknown_policy": "ignore"}, {"batch_size": 1}, Path(temp), {})
        self.assertEqual(measured["library_probability_metrics"]["f1_score"], .5)
        self.assertEqual(measured["argmax_pixel_metrics"]["included_pixels"], 3)
        self.assertEqual(measured["argmax_pixel_metrics"]["ignored_pixels"], 1)
        self.assertFalse(measured["historical_report_result_reproduced"])

    def test_all_ignored_batch_is_excluded_instead_of_rewarded(self):
        from types import SimpleNamespace
        batch = [{"image": np.zeros((1, 2, 3), dtype=np.uint8),
                  "mask": np.full((1, 2), 255, dtype=np.uint8)}]
        probs = np.eye(5, dtype=np.float32)[np.zeros((1, 1, 2), dtype=np.uint8)]
        with tempfile.TemporaryDirectory() as temp, patch("building_damage.scene_workflow.scene_batches", return_value=iter([batch])):
            measured = evaluate(lambda x, training=False: probs, SimpleNamespace(),
                                {"scenes": [1], "unknown_policy": "ignore"}, {"batch_size": 1}, Path(temp), {})
            progress = json.loads((Path(temp) / "progress.json").read_text())
        self.assertEqual(progress["status"], "completed")
        self.assertEqual(measured["fully_ignored_batches_excluded_from_library_means"], 1)
        self.assertIsNone(measured["library_probability_metrics"]["f1_score"])
        self.assertIsNone(measured["argmax_pixel_metrics"]["pixel_accuracy"])


if __name__ == "__main__":
    unittest.main()

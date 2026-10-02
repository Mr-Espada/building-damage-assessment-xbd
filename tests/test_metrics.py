"""Hand-calculated checks for new argmax metrics; no model or xBD data."""

import json
import unittest

import numpy as np

from building_damage.metrics import ConfusionMetrics


def probabilities(predicted_labels):
    return np.eye(5, dtype=np.float32)[np.asarray(predicted_labels)]


class PixelMetrics(unittest.TestCase):
    def test_hand_calculated_counts_scores_and_partition_invariance(self):
        # Rows: background [1,1,0], undamaged [0,1,1], minor [0,0,2].
        truth = np.array([[[0, 0, 1], [1, 2, 2]]], dtype=np.uint8)
        prediction = probabilities([[[0, 1, 1], [2, 2, 2]]])
        metrics = ConfusionMetrics()
        metrics.update(truth, prediction)
        result = metrics.result()
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 2, 0, 0], [0]*5, [0]*5])
        self.assertEqual(result["included_pixels"], 6)
        self.assertAlmostEqual(result["pixel_accuracy"], 4/6)
        classes = result["per_class"]
        self.assertAlmostEqual(classes[0]["iou"], 1/2)
        self.assertAlmostEqual(classes[0]["f1"], 2/3)
        self.assertEqual(classes[0]["precision"], 1.0)
        self.assertEqual(classes[0]["recall"], 0.5)
        self.assertAlmostEqual(classes[1]["iou"], 1/3)
        self.assertAlmostEqual(classes[1]["f1"], 1/2)
        self.assertAlmostEqual(classes[2]["iou"], 2/3)
        self.assertAlmostEqual(classes[2]["f1"], 4/5)
        self.assertAlmostEqual(result["macro_all_classes"]["iou"], 1/2)
        self.assertAlmostEqual(result["macro_building_classes"]["f1"], (1/2 + 4/5)/2)
        self.assertEqual(result["macro_all_classes"]["defined_class_ids"]["iou"], [0, 1, 2])
        split = ConfusionMetrics()
        for row in range(2):
            split.update(truth[0, row:row+1], prediction[0, row:row+1])
        self.assertEqual(split.result(), result)
        json.dumps(result, allow_nan=False)

    def test_absent_classes_and_false_positive_are_distinct(self):
        metrics = ConfusionMetrics()
        metrics.update(np.array([[0]], dtype=np.uint8), probabilities([[1]]))
        result = metrics.result()
        self.assertEqual(result["per_class"][0]["f1"], 0.0)
        self.assertIsNone(result["per_class"][0]["precision"])
        self.assertEqual(result["per_class"][1]["iou"], 0.0)
        self.assertEqual(result["per_class"][1]["precision"], 0.0)
        self.assertIsNone(result["per_class"][1]["recall"])
        self.assertIsNone(result["per_class"][2]["f1"])
        self.assertEqual(result["macro_all_classes"]["defined_class_ids"]["f1"], [0, 1])

    def test_ignored_pixels_and_all_ignored_run(self):
        metrics = ConfusionMetrics()
        metrics.update(np.array([[0, 255]], dtype=np.uint8), probabilities([[0, 4]]))
        result = metrics.result()
        self.assertEqual(result["included_pixels"], 1)
        self.assertEqual(result["ignored_pixels"], 1)
        self.assertEqual(result["pixel_accuracy"], 1.0)
        self.assertEqual(result["per_class"][4]["predicted_pixels"], 0)
        ignored = ConfusionMetrics()
        ignored.update(np.full((1, 1), 255, dtype=np.uint8), probabilities([[3]]))
        self.assertIsNone(ignored.result()["pixel_accuracy"])
        self.assertIsNone(ignored.result()["macro_all_classes"]["iou"])
        json.dumps(ignored.result(), allow_nan=False)

    def test_invalid_input_never_changes_counts(self):
        metrics = ConfusionMetrics()
        good_labels = np.array([[0]], dtype=np.uint8)
        good_probabilities = probabilities([[0]])
        bad_probabilities = good_probabilities.copy()
        bad_probabilities[0, 0, 0] = np.nan
        invalid_pairs = [
            (np.array([[5]], dtype=np.uint8), good_probabilities),
            (np.array([[-1]], dtype=np.int32), good_probabilities),
            (np.array([[0.5]]), good_probabilities),
            (np.array([[False]]), good_probabilities),
            (good_labels, bad_probabilities),
            (good_labels, np.zeros((1, 1, 5), dtype=np.float32)),
            (good_labels, np.array([[[1.1, -0.1, 0, 0, 0]]])),
            (good_labels, np.ones((1, 1, 4))),
            (np.zeros(1, dtype=np.uint8), np.ones((1, 5))/5),
            (np.zeros((0, 1), dtype=np.uint8), np.zeros((0, 1, 5))),
        ]
        for labels, values in invalid_pairs:
            with self.subTest(labels=labels.tolist(), shape=values.shape):
                with self.assertRaises(ValueError):
                    metrics.update(labels, values)
                self.assertEqual(metrics.result()["included_pixels"], 0)
                self.assertEqual(metrics.result()["ignored_pixels"], 0)

    def test_argmax_uses_probabilities_and_documented_tie_rule(self):
        metrics = ConfusionMetrics()
        metrics.update(np.array([[0, 2]], dtype=np.uint8), np.array([[[0.5, 0.5, 0, 0, 0], [0.1, 0.1, 0.6, 0.1, 0.1]]]))
        self.assertEqual(metrics.result()["pixel_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()

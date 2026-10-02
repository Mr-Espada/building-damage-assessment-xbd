"""Dataset-level argmax pixel metrics added for later local evaluation.

These counts are distinct from the historical Segmentation Models soft metrics.
No TensorFlow import is needed, and no official xView2 score is implemented.
"""

import numpy as np


CLASS_NAMES = ("background", "undamaged", "minor_damage", "major_damage", "destroyed")


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


class ConfusionMetrics:
    """Accumulate rows=ground-truth, columns=argmax-prediction counts.

    ``update`` accepts labels shaped HW or NHW and matching softmax probabilities
    shaped HW5 or NHW5. Label 255 is excluded from every score. Ties follow NumPy's
    argmax rule and select the lowest class ID.
    """

    def __init__(self, classes=5, ignore_label=255):
        if classes != 5 or ignore_label != 255:
            raise ValueError("This evaluation protocol uses five classes and ignore label 255.")
        self.classes = classes
        self.ignore_label = ignore_label
        self.confusion_matrix = np.zeros((classes, classes), dtype=np.int64)
        self.ignored_pixels = 0

    def update(self, labels, probabilities):
        labels = np.asarray(labels)
        probabilities = np.asarray(probabilities)
        if labels.ndim not in (2, 3) or not labels.size:
            raise ValueError("Labels must be a nonempty HW or NHW array.")
        if labels.dtype.kind not in "iu":
            raise ValueError("Labels must have integer dtype; fractional and boolean labels are invalid.")
        valid_labels = ((labels >= 0) & (labels < self.classes)) | (labels == self.ignore_label)
        if not np.all(valid_labels):
            raise ValueError("Labels must be class IDs 0-4 or ignored label 255.")
        if probabilities.shape != labels.shape + (self.classes,):
            raise ValueError("Probabilities must match label dimensions with five class channels.")
        if probabilities.dtype.kind not in "fiu" or not np.all(np.isfinite(probabilities)):
            raise ValueError("Probabilities must be finite numeric values.")
        if np.any(probabilities < 0) or np.any(probabilities > 1):
            raise ValueError("Probabilities must be within [0, 1].")
        if not np.allclose(probabilities.sum(axis=-1), 1.0, rtol=1e-5, atol=1e-4):
            raise ValueError("Each pixel's five probabilities must sum to one.")

        included = labels != self.ignore_label
        predictions = probabilities.argmax(axis=-1)
        indexes = labels[included].astype(np.int64) * self.classes + predictions[included]
        counts = np.bincount(indexes, minlength=self.classes**2).reshape(self.classes, self.classes)
        # All validation completes before any counters are changed.
        self.confusion_matrix += counts
        self.ignored_pixels += int((~included).sum())

    def result(self):
        """Return a JSON-safe snapshot; undefined denominators become None."""
        per_class = []
        for class_id, name in enumerate(CLASS_NAMES):
            true_positive = int(self.confusion_matrix[class_id, class_id])
            truth_pixels = int(self.confusion_matrix[class_id, :].sum())
            predicted_pixels = int(self.confusion_matrix[:, class_id].sum())
            false_positive = predicted_pixels - true_positive
            false_negative = truth_pixels - true_positive
            per_class.append({
                "class_id": class_id,
                "name": name,
                "truth_pixels": truth_pixels,
                "predicted_pixels": predicted_pixels,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "iou": _ratio(true_positive, true_positive + false_positive + false_negative),
                "f1": _ratio(2 * true_positive, 2 * true_positive + false_positive + false_negative),
                "precision": _ratio(true_positive, predicted_pixels),
                "recall": _ratio(true_positive, truth_pixels),
            })

        def macro(class_ids):
            result = {"class_ids": list(class_ids), "defined_class_ids": {}}
            for metric in ("iou", "f1", "precision", "recall"):
                defined = [class_id for class_id in class_ids if per_class[class_id][metric] is not None]
                result["defined_class_ids"][metric] = defined
                result[metric] = float(np.mean([per_class[class_id][metric] for class_id in defined])) if defined else None
            return result

        included_pixels = int(self.confusion_matrix.sum())
        return {
            "metric_family": "dataset_level_argmax_pixel_confusion",
            "official_xview2_score": False,
            "semantics": {
                "prediction": "argmax over five class probabilities; tied maxima use the lowest class ID",
                "confusion_matrix": "rows are ground-truth class IDs; columns are predicted class IDs",
                "aggregation": "sum pixel counts over all included pixels before calculating scores; independent of batch partitioning",
                "ignore_label": self.ignore_label,
                "undefined_scores": "null when the relevant denominator is zero; macro means exclude undefined classes separately for each metric",
                "pixel_accuracy": "correct pixels divided by included pixels; equals all-class micro precision, recall and F1 for single-label classification",
                "historical_comparison": "distinct from the original probability-based Segmentation Models IoU and F1",
            },
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class": per_class,
            "macro_all_classes": macro(range(self.classes)),
            "macro_building_classes": macro(range(1, self.classes)),
            "pixel_accuracy": _ratio(int(np.trace(self.confusion_matrix)), included_pixels),
            "included_pixels": included_pixels,
            "ignored_pixels": self.ignored_pixels,
        }

"""Synthetic checks for new wrappers. No xBD imagery, weights or training."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from building_damage.arrays import open_arrays, batch
from building_damage.__main__ import main, load_config


class ArrayContract(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, name, value):
        path = self.root / name
        np.save(path, value)
        return path

    def test_pairing_and_object_rejection(self):
        x = self.save("x.npy", np.zeros((2, 512, 512, 3), dtype=np.uint8))
        wrong_y = self.save("wrong_y.npy", np.zeros((1, 512, 512), dtype=np.uint8))
        with self.assertRaises(ValueError):
            open_arrays(x, wrong_y)
        object_x = self.save("object.npy", np.array([{"not": "an image"}], dtype=object))
        with self.assertRaises(ValueError):
            open_arrays(object_x, wrong_y)

    def test_normalization_one_hot_and_invalid_label(self):
        images = np.full((1, 2, 2, 3), 255, dtype=np.uint8)
        labels = np.array([[[0, 1], [3, 4]]], dtype=np.uint8)
        x, y = batch(images, labels, np.array([0]), 5)
        np.testing.assert_array_equal(x, np.ones_like(x))
        np.testing.assert_array_equal(y.argmax(axis=-1), labels)
        np.testing.assert_array_equal(y.sum(axis=-1), np.ones((1, 2, 2)))
        labels[0, 0, 0] = 5
        with self.assertRaises(ValueError):
            batch(images, labels, np.array([0]), 5)

    def test_plan_never_starts_run_and_rejects_reused_split(self):
        x = np.zeros((1, 512, 512, 3), dtype=np.uint8)
        y = np.zeros((1, 512, 512), dtype=np.uint8)
        paths = [self.save(n, a) for n, a in (("train_x.npy", x), ("train_y.npy", y), ("val_x.npy", x), ("val_y.npy", y))]
        repo = Path(__file__).resolve().parents[1]
        args = ["train", "--config", str(repo / "configs/report_described.json"), "--output-dir", str(self.root / "untouched_output")]
        for key, path in zip(("train-images", "train-masks", "val-images", "val-masks"), paths):
            args.extend(["--" + key, str(path)])
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertFalse((self.root / "untouched_output").exists())
        args[args.index("--val-images") + 1] = str(paths[0])
        with self.assertRaises(ValueError):
            main(args)

    def test_no_overwrite_and_invalid_config(self):
        repo = Path(__file__).resolve().parents[1]
        cfg = json.loads((repo / "configs/report_described.json").read_text())
        cfg["learning_rate"] = 0
        bad = self.root / "bad.json"
        bad.write_text(json.dumps(cfg))
        with self.assertRaises(ValueError):
            load_config(bad)
        args = ["train", "--config", str(repo / "configs/report_described.json"),
                "--output-dir", str(self.root), "--train-images", "missing.npy",
                "--train-masks", "missing.npy", "--val-images", "missing.npy",
                "--val-masks", "missing.npy"]
        with self.assertRaisesRegex(ValueError, "new output directory"):
            main(args)

    def test_evaluation_plan_requires_tensorflow_checkpoint_and_does_not_execute(self):
        x = self.save("x.npy", np.zeros((1, 512, 512, 3), dtype=np.uint8))
        y = self.save("y.npy", np.zeros((1, 512, 512), dtype=np.uint8))
        repo = Path(__file__).resolve().parents[1]
        prefix = self.root / "candidate.ckpt"
        args = ["evaluate", "--config", str(repo / "configs/recovered_builder.json"),
                "--output-dir", str(self.root / "untouched_eval"),
                "--images", str(x), "--masks", str(y), "--checkpoint", str(prefix)]
        with self.assertRaisesRegex(ValueError, "TensorFlow checkpoint"):
            main(args)
        # Header-only plan verifies presence, not checkpoint contents or compatibility.
        Path(str(prefix) + ".index").write_bytes(b"synthetic presence marker")
        Path(str(prefix) + ".data-00000-of-00001").write_bytes(b"synthetic presence marker")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertFalse((self.root / "untouched_eval").exists())


if __name__ == "__main__":
    unittest.main()

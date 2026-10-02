"""Explicit-path training/evaluation added during repository cleanup."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

from .arrays import open_arrays, batch, metadata


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["backbone"] != "resnet101" or cfg["classes"] != 5 or cfg["image_size"] != 512:
        raise ValueError("This wrapper supports only the recovered ResNet101 five-class 512-pixel model.")
    for key in ("batch_size", "epochs"):
        if not isinstance(cfg[key], int) or isinstance(cfg[key], bool) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if not isinstance(cfg["learning_rate"], (int, float)) or not math.isfinite(cfg["learning_rate"]) or cfg["learning_rate"] <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if cfg["encoder_weights"] not in ("imagenet", None):
        raise ValueError("encoder_weights must be imagenet or null.")
    if not isinstance(cfg["seed"], int) or isinstance(cfg["seed"], bool) or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer.")
    return cfg


def make_sequence(tf, arrays, cfg, *, shuffle):
    import numpy as np

    class Tiles(tf.keras.utils.Sequence):
        def __init__(self):
            self.indexes = np.arange(len(arrays[0]))
            self.rng = np.random.default_rng(cfg["seed"])
            if shuffle:
                self.rng.shuffle(self.indexes)

        def __len__(self):
            return math.ceil(len(self.indexes) / cfg["batch_size"])

        def __getitem__(self, i):
            indexes = self.indexes[i * cfg["batch_size"] : (i + 1) * cfg["batch_size"]]
            return batch(*arrays, indexes, cfg["classes"])

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.indexes)

    return Tiles()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] in ("prepare-scenes", "predict-scenes", "evaluate-scenes", "train-scenes"):
        from .scene_workflow import main as scene_main
        return scene_main(argv)
    parser = argparse.ArgumentParser(description="Recovered capstone workflows; defaults to input checks, never automatic training.")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Check final arrays; --execute starts a separately authorized new run.")
    evaluate = sub.add_parser("evaluate", help="Check arrays/checkpoint; --execute records a separate new evaluation.")
    for name, help_text in (
        ("prepare-scenes", "Create a post-disaster raw-scene manifest without copying imagery."),
        ("predict-scenes", "Generate fresh prediction figures from a raw-scene manifest."),
        ("evaluate-scenes", "Evaluate all raw-scene tiles under a documented new protocol."),
        ("train-scenes", "Prepare or explicitly launch new training from raw scenes, without augmentation."),
    ):
        sub.add_parser(name, help=help_text)
    for p in (train, evaluate):
        p.add_argument("--config", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--execute", action="store_true")
    for key in ("train-images", "train-masks", "val-images", "val-masks"):
        train.add_argument("--" + key, required=True)
    for key in ("images", "masks", "checkpoint"):
        evaluate.add_argument("--" + key, required=True)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Choose a new output directory; existing runs are never overwritten.")
    plan = {
        "status": "new_run_plan_not_historical_reproduction",
        "command": args.command,
        "config": cfg,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
    }
    if args.command == "train":
        arrays = open_arrays(args.train_images, args.train_masks, cfg["image_size"])
        val_arrays = open_arrays(args.val_images, args.val_masks, cfg["image_size"])
        if Path(args.train_images).resolve() == Path(args.val_images).resolve() or Path(args.train_masks).resolve() == Path(args.val_masks).resolve():
            raise ValueError("Training and validation must use distinct files; a scene-level manifest is still required.")
        plan["train"] = metadata(args.train_images, args.train_masks, *arrays)
        plan["validation"] = metadata(args.val_images, args.val_masks, *val_arrays)
    else:
        arrays = open_arrays(args.images, args.masks, cfg["image_size"])
        prefix = Path(args.checkpoint)
        if not Path(str(prefix) + ".index").is_file() or not list(prefix.parent.glob(prefix.name + ".data-*-of-*")):
            raise ValueError("Expected a TensorFlow checkpoint prefix with index and data shard(s), not a .pth file.")
        plan["evaluation"] = metadata(args.images, args.masks, *arrays)
        plan["checkpoint"] = str(prefix.resolve())
        plan["checkpoint_epoch_and_report_association"] = "unverified"
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("Headers checked. No run started. Use --execute only for a separately authorized experiment.")
        return 0

    from .model import framework, build_model, seed_framework
    tf, sm = framework()
    seed_framework(tf, cfg["seed"])
    plan["environment"] = {
        name: importlib.metadata.version(name)
        for name in ("tensorflow", "tf-keras", "segmentation-models", "numpy")
    }
    plan["status"] = "started_new_run_not_historical_reproduction"
    output.mkdir(parents=True, exist_ok=False)
    dump(output / "run.json", plan)
    if args.command == "train":
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = build_model(cfg, training=True)
        history = model.fit(
            make_sequence(tf, arrays, cfg, shuffle=True),
            validation_data=make_sequence(tf, val_arrays, cfg, shuffle=False),
            shuffle=False,
            epochs=cfg["epochs"],
            callbacks=[tf.keras.callbacks.ModelCheckpoint(
                filepath=str(output / "best_model.ckpt"), monitor="val_loss",
                save_best_only=True, save_weights_only=True,
            ), tf.keras.callbacks.CSVLogger(str(output / "epoch_history.csv"))],
        )
        model.save_weights(str(output / "last_model.ckpt"))
        dump(output / "history.json", {k: [float(x) if math.isfinite(float(x)) else None for x in v] for k, v in history.history.items()})
    else:
        model = build_model(cfg)
        status = model.load_weights(args.checkpoint)
        status.assert_existing_objects_matched()
        status.expect_partial()  # optimizer extras may be present; all model variables must match
        measured = model.evaluate(make_sequence(tf, arrays, cfg, shuffle=False), return_dict=True)
        dump(output / "measured_metrics.json", {
            "status": "new_evaluation_of_unverified_checkpoint",
            "samples": len(arrays[0]),
            "batch_size": cfg["batch_size"],
            "metrics": {k: float(v) if math.isfinite(float(v)) else None for k, v in measured.items()},
            "nonfinite_values_encoded_as_null": True,
            "historical_report_result_reproduced": False,
        })
    plan["status"] = "completed_new_run_not_historical_reproduction"
    plan["finished_utc"] = datetime.now(timezone.utc).isoformat()
    dump(output / "run.json", plan)
    return 0


if __name__ == "__main__":
    main()

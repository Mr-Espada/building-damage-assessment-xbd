"""Later, manifest-backed local workflows. These do not reconstruct the 2023 run."""

import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import numpy as np

from .scenes import create_manifest, load_manifest, load_scene, iter_tiles
from .metrics import ConfusionMetrics


CLASS_NAMES = ["Background", "Undamaged", "Minor damage", "Major damage", "Destroyed"]
PALETTE = np.array([[0, 0, 0], [51, 255, 255], [255, 255, 0],
                    [255, 128, 0], [255, 0, 0]], dtype=np.uint8)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def checkpoint_files(prefix):
    prefix = Path(prefix).resolve()
    files = [Path(str(prefix) + ".index"), *sorted(prefix.parent.glob(prefix.name + ".data-*-of-*"))]
    if len(files) < 2 or not all(p.is_file() for p in files):
        raise ValueError("Expected a TensorFlow checkpoint prefix with index and data shards.")
    return files


def targets(masks):
    """255 is ignored explicitly; background remains a scored class."""
    valid = masks != 255
    if np.any((masks[valid] < 0) | (masks[valid] > 4)):
        raise ValueError("Mask IDs must be 0..4 or the explicit ignore ID 255.")
    return np.eye(5, dtype=np.float32)[np.where(valid, masks, 0)] * valid[..., None]


def render_scene(path, rgb, truth, predicted, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    actual = PALETTE[np.where(truth == 255, 0, truth)].copy()
    actual[truth == 255] = [160, 160, 160]
    colors = PALETTE[predicted]
    overlay = rgb.copy()
    building = predicted != 0
    overlay[building] = (0.55 * rgb[building] + 0.45 * colors[building]).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, data, label in zip(axes, (rgb, actual, colors, overlay),
                               ("Post-disaster input", "Annotation mask", "Predicted classes", "Prediction overlay")):
        ax.imshow(data)
        ax.set_title(label)
        ax.axis("off")
    handles = [Patch(color=tuple(c / 255), label=n) for c, n in zip(PALETTE, CLASS_NAMES)]
    if np.any(truth == 255):
        handles.append(Patch(color=(160 / 255,) * 3, label="Ignored annotation"))
    fig.suptitle(title + "\nFresh local prediction; historical report association unverified", fontsize=10)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, .08, 1, .88))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def scene_batches(manifest, batch_size, source_records):
    """Scene-major tiles, with batches allowed to cross scene boundaries."""
    pending = []
    for entry in manifest["scenes"]:
        rgb, mask, provenance = load_scene(entry, manifest["unknown_policy"])
        source_records[f"{entry['split']}/{entry['scene_id']}"] = provenance
        for tile in iter_tiles(entry, rgb, mask):
            pending.append(tile)
            if len(pending) == batch_size:
                yield pending
                pending = []
    if pending:
        yield pending


def evaluate(model, sm, manifest, cfg, output, records):
    hard = ConfusionMetrics()
    sums = {"dice_loss": 0.0, "iou_score": 0.0, "f1_score": 0.0}
    batches = samples = scored_batches = scored_samples = fully_ignored_batches = 0
    began = time.monotonic()
    for tiles in scene_batches(manifest, cfg["batch_size"], records):
        x = np.stack([t["image"] for t in tiles]).astype(np.float32) / np.float32(255)
        masks = np.stack([t["mask"] for t in tiles])
        predicted = np.asarray(model(x, training=False))
        hard.update(masks, predicted)
        if np.any(masks != 255):
            y = targets(masks)
            masked_pred = predicted * (masks != 255)[..., None]
            measured = {"dice_loss": sm.losses.DiceLoss()(y, masked_pred),
                        "iou_score": sm.metrics.iou_score(y, masked_pred),
                        "f1_score": sm.metrics.f1_score(y, masked_pred)}
            for name, value in measured.items():
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite {name}; evaluation stopped.")
                # Keras loss uses sample weighting; library metric scalars are batch means.
                sums[name] += value * (len(tiles) if name == "dice_loss" else 1)
            scored_samples += len(tiles)
            scored_batches += 1
        else:
            # Literal smooth/smooth would reward a batch with no scored pixels.
            fully_ignored_batches += 1
        samples += len(tiles)
        batches += 1
        if batches == 1 or batches % 10 == 0:
            progress = {"status": "in_progress_not_final_metrics", "tiles_completed": samples,
                        "tiles_planned": 4 * len(manifest["scenes"]),
                        "elapsed_seconds": round(time.monotonic() - began, 2)}
            dump(output / "progress.json", progress)
            print(f"Evaluated {samples}/{4 * len(manifest['scenes'])} tiles", flush=True)
    dump(output / "progress.json", {"status": "completed", "tiles_completed": samples,
                                   "tiles_planned": 4 * len(manifest["scenes"]),
                                   "elapsed_seconds": round(time.monotonic() - began, 2)})
    return {"status": "new_evaluation_of_unverified_checkpoint", "scenes": len(manifest["scenes"]),
            "tiles": samples, "batches": batches, "batch_size": cfg["batch_size"],
            "elapsed_seconds": round(time.monotonic() - began, 2),
            "library_scored_batches": scored_batches, "library_scored_samples": scored_samples,
            "fully_ignored_batches_excluded_from_library_means": fully_ignored_batches,
            "library_probability_metrics": {name: (value / (scored_samples if name == "dice_loss" else scored_batches)) if scored_batches else None
                                            for name, value in sums.items()},
            "library_metric_semantics": "Segmentation Models defaults, smooth=1e-5, all five classes, no threshold; IoU/F1 arithmetic mean of batch scalars; Dice loss weighted by batch sample count. Ignored pixels zeroed in truth and prediction; batches with no scored pixels excluded. Batch size matters. Absent classes retain the library's smoothing behavior.",
            "argmax_pixel_metrics": hard.result(), "unknown_policy": manifest["unknown_policy"],
            "historical_report_result_reproduced": False,
            "official_xview2_score": False}


def predict(model, manifest, cfg, output, records, save_probabilities):
    figure_dir = output / "figures"
    figure_dir.mkdir()
    predictions = output / "predictions"
    predictions.mkdir()
    for entry in manifest["scenes"]:
        rgb, mask, provenance = load_scene(entry, manifest["unknown_policy"])
        records[f"{entry['split']}/{entry['scene_id']}"] = provenance
        tiles = list(iter_tiles(entry, rgb, mask))
        classes = np.zeros((1024, 1024), dtype=np.uint8)
        probs = np.zeros((1024, 1024, 5), dtype=np.float32) if save_probabilities else None
        for start in range(0, 4, cfg["batch_size"]):
            selected = tiles[start:start + cfg["batch_size"]]
            x = np.stack([t["image"] for t in selected]).astype(np.float32) / np.float32(255)
            values = np.asarray(model(x, training=False))
            # Reuse probability/label contract checks without reporting sample metrics.
            ConfusionMetrics().update(np.stack([t["mask"] for t in selected]), values)
            for tile, value in zip(selected, values):
                r, c = tile["row"], tile["col"]
                classes[r:r + 512, c:c + 512] = value.argmax(-1).astype(np.uint8)
                if probs is not None:
                    probs[r:r + 512, c:c + 512] = value
        stem = entry["split"] + "__" + entry["scene_id"]
        render_scene(figure_dir / (stem + ".png"), rgb, mask, classes, stem)
        np.save(predictions / (stem + "_classes.npy"), classes)
        if probs is not None:
            np.save(predictions / (stem + "_probabilities.npy"), probs)
        print(f"Saved figure for {stem}", flush=True)


def training_sequence(tf, manifest, cfg, records, shuffle):
    class Tiles(tf.keras.utils.Sequence):
        def __init__(self):
            self.indexes = np.arange(4 * len(manifest["scenes"]))
            self.rng = np.random.default_rng(cfg["seed"])
            if shuffle:
                self.rng.shuffle(self.indexes)

        @lru_cache(maxsize=4)
        def scene(self, i):
            entry = manifest["scenes"][i]
            rgb, mask, provenance = load_scene(entry, manifest["unknown_policy"])
            records[f"{entry['split']}/{entry['scene_id']}"] = provenance
            return list(iter_tiles(entry, rgb, mask))

        def __len__(self):
            return math.ceil(len(self.indexes) / cfg["batch_size"])

        def __getitem__(self, i):
            ids = self.indexes[i * cfg["batch_size"]:(i + 1) * cfg["batch_size"]]
            tiles = [self.scene(int(index) // 4)[int(index) % 4] for index in ids]
            x = np.stack([t["image"] for t in tiles]).astype(np.float32) / np.float32(255)
            return x, targets(np.stack([t["mask"] for t in tiles]))

        def on_epoch_end(self):
            if shuffle:
                self.rng.shuffle(self.indexes)
    return Tiles()


def validate_training_manifests(train, validation):
    # Full scene IDs embed the disaster and scene number, independent of split.
    train_ids = {e["scene_id"] for e in train["scenes"]}
    val_ids = {e["scene_id"] for e in validation["scenes"]}
    train_paths = {e["image"] for e in train["scenes"]}
    val_paths = {e["image"] for e in validation["scenes"]}
    if train_ids & val_ids or train_paths & val_paths:
        raise ValueError("Training and validation overlap at scene level.")
    if train["unknown_policy"] != validation["unknown_policy"]:
        raise ValueError("Training and validation must use the same unknown-label policy.")


def main(argv):
    parser = argparse.ArgumentParser(description="Fresh local xBD workflows; originals remain unchanged.")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-scenes", help="Write a metadata-only post-disaster scene manifest, without copying images.")
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--splits", nargs="+", choices=("tier1", "tier3", "hold", "test"), required=True)
    prepare.add_argument("--scene", action="append", dest="scene_ids")
    prepare.add_argument("--unknown-policy", choices=("error", "ignore", "undamaged_legacy"), default="error")
    prepare.add_argument("--output-manifest", required=True)
    for name in ("predict-scenes", "evaluate-scenes", "train-scenes"):
        p = sub.add_parser(name)
        p.add_argument("--manifest", required=True)
        p.add_argument("--config", required=True)
        p.add_argument("--batch-size", type=int)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--execute", action="store_true")
        if name == "train-scenes":
            p.add_argument("--validation-manifest", required=True)
        else:
            p.add_argument("--checkpoint", required=True)
        if name == "predict-scenes":
            p.add_argument("--save-probabilities", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare-scenes":
        path = Path(args.output_manifest)
        if path.exists():
            raise ValueError("Choose a new manifest path; existing manifests are not overwritten.")
        value = create_manifest(args.dataset_root, args.splits, args.unknown_policy, args.scene_ids)
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(path, value)
        print(f"Prepared {len(value['scenes'])} post-disaster scenes / {4 * len(value['scenes'])} tiles; no image data copied.")
        return 0

    from .__main__ import load_config
    cfg = load_config(args.config)
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("batch-size must be positive.")
        cfg["batch_size"] = args.batch_size
    manifest = load_manifest(args.manifest)
    output = Path(args.output_dir)
    if output.exists():
        raise ValueError("Choose a new output directory; existing runs are never overwritten.")
    plan = {"status": "new_run_plan_not_historical_reproduction", "command": args.command,
            "config": cfg, "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": digest(args.manifest), "scene_count": len(manifest["scenes"]),
            "tile_count": 4 * len(manifest["scenes"]), "protocol": manifest["protocol"],
            "unknown_policy": manifest["unknown_policy"], "normalization": "float32 RGB / 255",
            "checkpoint_epoch_and_report_association": "unverified"}
    plan["runner_sources_sha256"] = {name: digest(Path(__file__).parent / name) for name in
                                     ("__main__.py", "scene_workflow.py", "scenes.py", "metrics.py", "model.py")}
    plan["config_file_sha256"] = digest(args.config)
    validation = None
    if args.command == "train-scenes":
        validation = load_manifest(args.validation_manifest)
        validate_training_manifests(manifest, validation)
        plan["validation_manifest"] = str(Path(args.validation_manifest).resolve())
        plan["validation_manifest_sha256"] = digest(args.validation_manifest)
        plan["validation_scenes"] = len(validation["scenes"])
        plan["training_method"] = "Fresh initialization, all four tiles, no augmentation; NOT the missing 186404-tile historical preparation or a resume."
    else:
        checkpoint_files(args.checkpoint)
        plan["checkpoint"] = str(Path(args.checkpoint).resolve())
    if not args.execute:
        print(json.dumps(plan, indent=2))
        print("Plan only. Add --execute to start this explicitly selected new run.")
        return 0

    output.mkdir(parents=True, exist_ok=False)
    dump(output / "manifest.json", manifest)
    if validation is not None:
        dump(output / "validation_manifest.json", validation)
    plan["status"] = "started_new_run_not_historical_reproduction"
    dump(output / "run.json", plan)
    records = {}
    try:
        from .model import framework, build_model, seed_framework
        tf, sm = framework()
        seed_framework(tf, cfg["seed"])
        plan["devices"] = {"physical": [{"name": d.name, "type": d.device_type} for d in tf.config.list_physical_devices()],
                           "logical": [{"name": d.name, "type": d.device_type} for d in tf.config.list_logical_devices()]}
        plan["environment"] = {name: importlib.metadata.version(name) for name in
                               ("tensorflow", "tf-keras", "segmentation-models", "numpy", "tifffile", "shapely", "opencv-python-headless", "matplotlib")}
        if args.command == "train-scenes":
            with tf.distribute.MirroredStrategy().scope():
                model = build_model(cfg, training=True)
                # One-hot targets are all zero only for explicitly ignored pixels.
                def mask_call(function, name):
                    def masked(y_true, y_pred):
                        valid = tf.reduce_sum(y_true, axis=-1, keepdims=True)
                        return function(y_true, y_pred * valid)
                    masked.__name__ = name
                    return masked
                model.compile(optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
                              loss=mask_call(sm.losses.DiceLoss(), "masked_dice_loss"),
                              metrics=[mask_call(sm.metrics.iou_score, "iou_score"), mask_call(sm.metrics.f1_score, "f1_score")])
            dump(output / "run.json", plan)
            history = model.fit(training_sequence(tf, manifest, cfg, records, True),
                                validation_data=training_sequence(tf, validation, cfg, records, False),
                                epochs=cfg["epochs"], shuffle=False,
                                callbacks=[tf.keras.callbacks.ModelCheckpoint(str(output / "best_model.ckpt"),
                                           monitor="val_loss", save_best_only=True, save_weights_only=True),
                                           tf.keras.callbacks.CSVLogger(str(output / "epoch_history.csv"))])
            model.save_weights(str(output / "last_model.ckpt"))
            dump(output / "history.json", {k: [float(v) if math.isfinite(float(v)) else None for v in values]
                                           for k, values in history.history.items()})
        else:
            plan["checkpoint_files"] = [{"path": str(p), "bytes": p.stat().st_size, "sha256": digest(p)}
                                         for p in checkpoint_files(args.checkpoint)]
            dump(output / "run.json", plan)
            model = build_model(cfg)
            restored = model.load_weights(args.checkpoint)
            restored.assert_existing_objects_matched()
            restored.expect_partial()
            if args.command == "predict-scenes":
                predict(model, manifest, cfg, output, records, args.save_probabilities)
            else:
                dump(output / "measured_metrics.json", evaluate(model, sm, manifest, cfg, output, records))
        plan["status"] = "completed_new_run_not_historical_reproduction"
    except BaseException as error:
        plan["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        plan["error"] = f"{type(error).__name__}: {error}"
        if (output / "progress.json").exists():
            progress = json.loads((output / "progress.json").read_text())
            progress["status"] = plan["status"]
            dump(output / "progress.json", progress)
        raise
    finally:
        plan["finished_utc"] = datetime.now(timezone.utc).isoformat()
        dump(output / "source_provenance.json", records)
        dump(output / "run.json", plan)
    return 0

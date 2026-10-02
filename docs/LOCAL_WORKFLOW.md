# Local prediction, evaluation and future training

The October 2026 scene workflow can use existing xBD TIFF/JSON pairs and a compatible local TensorFlow checkpoint without creating or loading huge processed arrays. It streams complete post-disaster scenes, makes all four 512×512 crops, and keeps empty tiles. It performs no resizing, resampling, augmentation or image download. This is a new, documented preparation protocol, not recovery of the missing historical arrays or the 105-epoch run.

Install the package as described in [ENVIRONMENT.md](ENVIRONMENT.md). Keep dataset originals, checkpoint files, manifests and run outputs outside the repository. Use a dataset root containing:

```text
/external/xbd/
  tier1/{images,labels}/
  tier3/{images,labels}/
  hold/{images,labels}/
  test/{images,labels}/
```

Each scene pairs `images/<scene>_post_disaster.tif` with `labels/<scene>_post_disaster.json`. Scene IDs are full filename stems, including `_post_disaster`. An example ID is `hurricane-harvey_00000127_post_disaster`; substitute an available scene when selecting an example. Pass a checkpoint prefix such as `/external/checkpoints/best_model.ckpt`, with its `.index` and `.data-*-of-*` files alongside it. Weights remain local.

## Prepare a scene manifest

```bash
python -m building_damage prepare-scenes \
  --dataset-root /external/xbd --splits test \
  --unknown-policy ignore \
  --output-manifest /external/capstone-work/manifests/test.json
```

Manifest creation reads filenames and file metadata, checks matching post-disaster image/annotation names, and writes a deterministic scene list. It does not copy images or inspect their payloads. Actual loading verifies recorded file sizes/timestamps, reads one scene, hashes image/annotation contents, and checks their dimensions and types.

The default unknown-label policy is `error`: execution stops on an unrecognized or missing damage subtype. The example chooses `ignore`, recommended for fresh testing: such building pixels become 255 and are excluded from loss and scores. `undamaged_legacy` maps them to class 1, reproducing the old helper's subtype fallback. Policy choice changes the evaluation and must be reported. Background remains class 0 and is scored under every policy.

TIFFs must be integer RGB arrays with shape `(1024,1024,3)` and values within `[0,255]`. This accepts range-checked local int16 TIFFs, then converts them to uint8 without wrapping. Floating-point, out-of-range, wrongly sized or differently arranged inputs fail explicitly. Annotation polygons use pixel coordinates truncated toward zero; holes are preserved and later annotations win on overlaps. Tile order is top-left, top-right, bottom-left, bottom-right; normalization is float32 RGB `/255` at batch loading.

## Generate prediction figures

Select complete scenes for figures rather than generating thousands of plots unintentionally:

```bash
python -m building_damage prepare-scenes \
  --dataset-root /external/xbd --splits test \
  --scene hurricane-harvey_00000127_post_disaster \
  --unknown-policy ignore \
  --output-manifest /external/capstone-work/manifests/figures.json

python -m building_damage predict-scenes \
  --manifest /external/capstone-work/manifests/figures.json \
  --config configs/local_inference.json \
  --checkpoint /external/checkpoints/best_model.ckpt \
  --output-dir /external/capstone-work/runs/prediction-001
```

The prediction command above prints a plan. Add `--execute` to run it. Repeat `--scene` for additional specified scenes. Each output figure shows the post-disaster input, annotation mask, predicted classes and prediction overlay, with the historical class colors and an explicit fresh-prediction label. Class-ID arrays are saved separately; `--save-probabilities` additionally saves full five-channel float32 probabilities, increasing output size.

`configs/local_inference.json` defaults to batch size 1 for modest memory use. `--batch-size` overrides it. The learning-rate/epoch placeholders in this config do not affect inference; no ImageNet initialization download is used when restoring weights. Full ResNet101 inference can still be slow on CPU, so start with an explicitly selected scene.

## Evaluate a selected split

```bash
python -m building_damage evaluate-scenes \
  --manifest /external/capstone-work/manifests/test.json \
  --config configs/local_inference.json \
  --checkpoint /external/checkpoints/best_model.ckpt \
  --output-dir /external/capstone-work/runs/test-evaluation-001
```

Inspect the plan, then add `--execute` for a new evaluation. Prepare a separate `--splits hold` manifest for hold results. A deliberately chosen `--splits hold test` manifest combines both; this resembles the report's split selection but does not reproduce its unavailable preprocessing/evaluation record. The retained checkpoint may already have been selected using hold+test, so evaluating it on test cannot establish a previously untouched independent test result.

`measured_metrics.json` contains two distinct score families:

- `library_probability_metrics`: Segmentation Models Dice loss, IoU and F1 with default smoothing `1e-5`, all five classes including background and no probability threshold. Ignored pixels are zeroed in both targets and predictions. IoU/F1 are averaged across batch scalars; Dice loss is weighted by batch sample count. Batch size and scene order are recorded and can affect these values.
- `argmax_pixel_metrics`: one confusion matrix accumulated across all included pixels, with rows as truth and columns as prediction. It reports per-class counts, IoU, F1, precision and recall; macros over classes 0–4 and separately 1–4; and pixel accuracy. Undefined denominators produce `null`, with macro means excluding undefined classes separately for each metric. These counts are independent of batch partitioning for fixed predictions.

Neither family is the official xView2 competition score. Every measured result states that historical report reproduction is false. Prediction-only figures do not report test performance, and a selected scene subset cannot substitute for a complete split evaluation. Keep the original report's IoU 0.62743 and F1 0.6746 unchanged and label any fresh values separately.

## Prepare future training

No training starts from the examples below. For a later explicitly approved experiment, use tier1+tier3 for training, hold for validation and reserve test from selection/tuning:

```bash
python -m building_damage prepare-scenes \
  --dataset-root /external/xbd --splits tier1 tier3 \
  --unknown-policy ignore \
  --output-manifest /external/capstone-work/manifests/train.json

python -m building_damage prepare-scenes \
  --dataset-root /external/xbd --splits hold \
  --unknown-policy ignore \
  --output-manifest /external/capstone-work/manifests/hold.json

python -m building_damage train-scenes \
  --manifest /external/capstone-work/manifests/train.json \
  --validation-manifest /external/capstone-work/manifests/hold.json \
  --config configs/report_described.json \
  --output-dir /external/capstone-work/runs/new-training-001
```

The training plan checks scene-level overlap and requires identical unknown policies. It retains all four tiles without augmentation. This differs from the report's damage-targeted augmentation, empty training-tile removal, 186,404-tile total and combined hold+test validation. Event/geographic independence is not established by the scene overlap check.

Future execution requires `--execute`, a deliberately reviewed training configuration and adequate hardware. This starts fresh initialization; it does not resume the retained checkpoint or repair its epoch counter. The report config requests ImageNet initialization and 150 planned epochs, which are choices for this future run rather than proof of the historical parameters. Training writes epoch history, best validation-loss weights and last weights.

## Run records and preservation

Every manifest/output path must be new; existing manifests and runs are never overwritten. Plan-only prediction/evaluation/training creates no output directory and does not load the checkpoint or image payloads. Manifest preparation does write its small metadata file.

Executed runs save copied manifests and `run.json`: command/configuration, effective batch size, normalization, protocol, unknown policy, scene/tile counts, manifest SHA-256, runtime package versions and checkpoint paths/byte sizes/SHA-256. Training also records its validation manifest hash. `source_provenance.json` records loaded scene image/label hashes, source dtype/shape, annotation subtype counts, unknown counts and rasterization method. Evaluation writes a progress record during execution, then final metrics only after evaluation completes. Source records describe only scenes actually read.

The final run status distinguishes completed, failed and interrupted runs and records an error when applicable. Outputs left by an interrupted prediction or training run are partial artifacts; they do not establish a completed experiment. Use a new output path on retry. Input metadata checks catch ordinary file changes; the content hashes recorded during loading identify the payloads actually used, rather than authenticating the old HPC run.

Keep a personal profile or launcher containing your machine's paths outside Git, for example `/external/capstone-work/local-profile.json`. It should identify the dataset root, checkpoint prefix, output root and chosen config. Public commands use generic paths; no weights, private path profile or generated data/figures need to be committed. Original undergraduate scripts, notebooks, data and checkpoints remain preserved separately. See [CHANGELOG.md](../CHANGELOG.md) for the distinction between undergraduate work and these later additions.

Entirely ignored batches are excluded from soft-score means and counted explicitly; literal library smoothing would otherwise assign IoU/F1=1 and loss=0 to a batch with no scored pixels. Hard scores return null when denominators are absent. Runner source/config hashes and TensorFlow devices are recorded. The Python3.12 compatibility wrapper seeds Python, NumPy and TensorFlow directly to avoid a legacy Keras seeded-initializer bug; it does not promise bitwise GPU determinism.

The retained historical candidate may have been selected using hold+test validation. A new score on the subset named test therefore does not establish an independent held-out benchmark for that candidate. Reserving test applies to the proposed future training protocol.

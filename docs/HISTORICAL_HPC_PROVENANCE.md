# Historical HPC training and checkpoint provenance

Second-pass forensic review, 1 October 2026. This review supplements the first audit; it does not change the undergraduate results or authenticate a missing training run.

**Verdict: Plausible final-checkpoint candidate.** `27112023184806/best_model.ckpt` is the strongest recovered candidate for the final HPC-trained model described in the report. Its selection for historical local inference is strongly supported by notebook records. Its HPC origin, association with the report's 105-epoch run, and validation IoU **0.62743** / F1 **0.6746** remain unverified.

## Evidence tiers

| Tier | Finding | Limit |
| --- | --- | --- |
| Verified artifact evidence | Both checkpoint candidates match the recovered five-class U-Net/ResNet101 structure. Both inference notebook sources explicitly select candidate06. The hidden notebook preserves sequential prediction calls and four PNG outputs. | Verified contents of recovered files; notebook outputs do not authenticate the current checkpoint bytes or the historical execution environment. |
| Author recollection | The author recalls training the final model on HPC, losing remote logs/metric records, retaining its weights and subsequently using them locally. | Recollection supplied in October 2026; not an independent run record. |
| Strong inference | Candidate06 was the intended checkpoint for the retained local inference workflow and was likely used in a saved execution. It is a better candidate than candidate05. | Relative paths, editable notebook source and potentially stale outputs prevent an immutable load-to-bytes association. |
| Weak inference | Timestamp-shaped directory names, 2023 modification dates and optimizer counts fit a late-2023 training history. Best-only selection could reconcile an earlier saved state with a 105-epoch run. | These do not identify the training machine, establish a shared run, prove an epoch, or assign report scores to a checkpoint. |
| Unresolved | Exact final driver/environment, optimizer continuity, final split/arrays, full history, selected epoch, and report prediction provenance. | No independent score reproduction is claimed. |

## Search scope and recovered lineage

The original pre-cleanup inventory covers 157,044 files and 75 directories. The second pass searched those paths and 108 ordinary source/config/text/checkpoint-state files, including 34 notebooks and hidden autosaves. It examined cell sources, saved text, errors, execution counts and kernel metadata. Static ASCII strings from 32 bytecode files were inspected without executing them. The figure comparison covered 26 notebook PNG outputs, 11 graph images, five standalone illustrations and five extracted report figure images. Raw dataset images/labels were not opened for this review.

The authoritative source is the Masters project's 42-page `Final Report.pdf`, SHA-256 `7f3852f5fa2f5897f1f4cc2de7c08a7064a20e06e1fc540d0753234c0466fdf3`. Hardware/training statements appear on printed page 22 / PDF page 31; results on printed page 24 / PDF page 33; the interruption is discussed in Appendix D. These are report statements, not recovered machine logs.

Notebook cells below are **one-based**, counting every cell in the original JSON. Public notebook copies have outputs removed; the exact originals and forensic derivatives remain in the separate local audit/archive.

- `Model_Tensorflow/model_testing.ipynb`, cell 4, builds the model and selects `27112023184806/best_model.ckpt`. `model.py` lines 18–20 call `model.load_weights(path)`.
- Hidden `.ipynb_checkpoints/model_testing-checkpoint.ipynb` has the same 30 cell sources and kernel metadata. Cell 4 has execution count 3 and no saved error; subsequent prediction/display cells retain increasing counts 4–14. Cells 10, 13, 16 and 20 contain PNG prediction/overlay outputs. This is additional evidence beyond the visible notebook's saved missing-library failure. The ImageNet encoder download belongs to model initialization; it is not evidence of downloading the HPC checkpoint.
- The visible notebook's load/build cell records `ModuleNotFoundError: segmentation_models`, and later predictions are cleared. A later broken local environment is compatible with the historical account, but individual cell execution dates cannot be recovered.
- The hidden notebook's first-100 evaluation output stops at `9/13` progress with no completed result. Its first-100/200 unsorted sampling, inclusion of pre-disaster images and mask resizing cannot verify the report. Neither procedure was executed during this review.
- No inference source selects candidate05. No stronger final TensorFlow checkpoint was recovered. `.pth` files and the extracted `best_model/` tree are alternative PyTorch artifacts, not a final TensorFlow SavedModel.

## Candidate states and optimizer interpretation

| Recovered state | Candidate05: `27112023184805` | Candidate06: `27112023184806` |
| --- | ---: | ---: |
| Checkpoint entries, including object graph | 1,244 | 1,244 |
| Variable tensors, excluding object graph | 1,243 | 1,243 |
| Model parameters | 51,606,046 | 51,606,046 |
| Optimizer iterations | 180,606 | 600,078 |
| Optimizer learning-rate value | approximately 1e-5 | approximately 1e-5 |
| Explicit epoch/history variable | Absent | Absent |
| Named by recovered inference code | No | Yes |

Both object graphs and variable names/shapes match. All 563 model tensors and all 678 optimizer-slot tensors differ between candidates. They are distinct saved states of the same structure, not duplicate weight copies. This does not prove that candidate06 resumed from candidate05 or that they belong to the same run. Checksums are in [checkpoint_candidates.json](../results/checkpoint_candidates.json).

The only scalar variables found were the optimizer's iteration counter and learning rate. There is no explicit epoch or history record. Under **all** of these assumptions—186,404 training samples, global batch 32, ceiling batching without dropping the last batch, one optimizer update per batch, a counter starting at zero, and no counter reset or preceding training—the arithmetic is:

```text
ceil(186404 / 32) = 5826 updates per assumed epoch
180606 = 31 × 5826
600078 = 103 × 5826
```

These are conditional epoch equivalents, not discovered epoch metadata. A resumed counter can span training sessions; a fresh optimizer or explicit reset can omit earlier updates. Different cardinality, `steps_per_epoch`, batching or update behavior changes the interpretation. Four GPU replicas do not by themselves multiply one synchronized optimizer update into four counter increments. The retained script batches before entering `MirroredStrategy`, consistent with global batch 32, but the final driver is missing.

TensorFlow checkpoints can include tracked optimizer variables alongside weights. This is observed in both candidates despite the retained recipe's `save_weights_only=True`. Restoring compatible optimizer state can preserve its counter and Adam slots; creating a fresh optimizer can start a new counter. No final-related resume, `initial_epoch`, counter-reset or optimizer-manipulation code was recovered. For inference with the same restored model variables, resetting optimizer state does not change predictions; it matters if training resumes. See [TensorFlow checkpoint semantics](https://www.tensorflow.org/guide/checkpoint).

The saved LR is a snapshot, **not proof of the initial learning rate** or of a scheduler. The report describes static LR 1e-4; retained builders default to 1e-5. No recovered final scheduler explains the difference, so the configuration association remains unresolved.

## Could a run ending at 105 retain an earlier checkpoint?

Yes, in principle. `Model_Tensorflow/Uncertain_logs/train.ipy` lines 55–62 uses a fixed `best_model.ckpt`, monitors `val_loss`, and sets `save_best_only=True`, `save_weights_only=True`. Such a callback retains the best monitored checkpoint; later completed epochs need not overwrite it. See [Keras ModelCheckpoint documentation](https://keras.io/2/api/callbacks/model_checkpoint/).

A run reaching 105 completed epochs could therefore retain a checkpoint at the conditional 103-epoch state if later validation loss did not improve. **No recovered history proves that this happened.** A best-loss checkpoint also need not have the IoU/F1 recorded for a final epoch; the mechanism cannot assign the report's numbers to the recovered weights.

The retained recipe is not the missing final run: it requests 10 epochs, truncates validation to 1,866 array entries, references absent arrays and calls an ambiguous `build_model(weights)` signature. Its timestamp path is `model_checkpoints/{time_string}/best_model.ckpt`; the recovered directories lack that parent. No transfer manifest explains the changed layout.

## Timeline and HPC environment

| Artifact | Current mtime, UTC | Interpretation |
| --- | --- | --- |
| Candidate05 shard / index / state | 29 Nov 2023, 07:20:16 / :22 / :24 | Predates report; unverified original-machine time. |
| Candidate06 shard / index / state | 1 Dec 2023, 23:28:54 / :56 / :56 | Predates report and is later than candidate05. |
| Authoritative report creation metadata | 12 Dec 2023, 14:09:10 (15:09:10 at UTC+01) | PDF metadata, not a training record. |
| Current `model.py` | 28 Jan 2025, 22:39:18 | Later filesystem artifact; no version history identifies training-time source. |
| Hidden / visible inference notebook | 4 Mar 2025, 16:11:12 / 5 Mar 2025, 16:46:57 | Later than report; does not date individual saved outputs. |

Folder names fit the script's `%d%m%Y%H%M%S` format: 27 November 2023, 18:48:05/06, original timezone unknown. This weak naming correspondence does not establish actual start times. Current checkpoint ctimes are in January 2025 and ownership is local. ctime records metadata changes, not file creation or an HPC transfer. No source machine, copying method or remote owner can be inferred reliably.

The report states four A40 GPUs. The retained script contains generic `MirroredStrategy`; no device list/count, hostname or scheduler is specified. The inference kernel identifies Python 3.8.18, not the complete training environment. No final-run SLURM/PBS jobs, scheduler output, A40 references, environment-module commands, conda/pip export, TensorBoard events, transfer commands or remote training paths were recovered from the source/path search.

The remote-looking CEDAR/axons path and saved cuDNN 8500 output belong to an unrelated 2022 ResNet50, four-class, 224-pixel microscopy template. Later SAM CUDA errors and local Torch paths are also unrelated. Neither supplies the final HPC environment. These scoped absence findings do not disprove the author's recollection of HPC training.

## Qualitative report figures

The hidden notebook supports a documentary chain from the **selected checkpoint path** to saved testing calls and outputs. No immutable checkpoint digest was recorded at execution, and no complete saved prediction artifact matches report Figures 13–17.

- Figure 13 and hidden cell 10 show strongly corresponding building/site geometry. The saved notebook selects `hurricane-harvey_00000127_pre_disaster.tif` and a tile; its dry imagery and undamaged masks differ from the report's flooded scene and major-damage masks. A post-disaster counterpart is a plausible scene identification, not a verified filename or identical prediction.
- Figure 17's source image and actual-mask pattern correspond to `graphs/all_masks.png`. That graph has no prediction panel or checkpoint reference. Training-set membership remains a report statement because the split manifest is absent.
- Figures 14–16 have no matched saved prediction output. Similar Image/Predicted/Actual layouts establish plotting-style correspondence only.

Thus **current checkpoint bytes → historical prediction artifact → report figure** is not verified. Partial scene/annotation links cannot validate the reported metrics. Private comparison measurements concern image-artifact alignment, not model performance, and are not new results.

## Publication and future evidence

An honest archival release can proceed with these disclosures; recovering the lost logs or retraining is not a prerequisite to that limited claim. The author subsequently confirmed code ownership and authorized the MIT license on 1 October 2026. The prepared code now includes that license and retains dependency attribution; publication should use the reviewed public Git file set. Do not add the private archive, data, weights or forensic image comparisons. Review separate rights before adding the report, satellite panels or external earthquake illustrations; the latter include BBC-branded and unattributed material.

Recovering an original final history, validation scene/tile manifest, environment record, or contemporaneous checkpoint digest would strengthen provenance. Any newly authorized training/evaluation must produce separate results. The reported values remain unchanged and not independently reproduced. See [reproducibility status](REPRODUCIBILITY.md), [publication checklist](RELEASE_CHECKLIST.md) and [safe application wording](APPLICATION_WORDING.md).

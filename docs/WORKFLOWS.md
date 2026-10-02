# Training and evaluation workflows

The array commands below were added during initial October 2026 repository preparation. At that phase no model was trained and no original-data evaluation was run. Both array commands default to checking inputs and printing a plan. `--execute` is a deliberate launch flag for a separately authorized experiment.

Inputs must be numeric, nonempty, paired uint8 NumPy arrays with shapes `(N,512,512,3)` and `(N,512,512)`. Object/pickle arrays are rejected. Every batch checks label range 0–4. Arrays are memory mapped and batches are normalized `/255` then masks are one-hot encoded. This avoids loading the full processed dataset into RAM. A data plan checks headers only; it does not inspect every label value until batches are read.

The report config uses five classes, 512-pixel input, batch 32, Adam 1e-4, Dice loss, Segmentation Models IoU/F1 and 150 planned epochs. The recovered-builder config uses 1e-5. Both state that the historical run association is unverified. No default validation truncation is applied. Training uses MirroredStrategy and a seed for initialization/shuffling; exact determinism across GPU configurations is not guaranteed.

Output directories must be new. Execution writes configuration, array metadata, environment versions, checkpoint path and a JSON run record before running. Training writes epoch history and best/last TensorFlow weights. Evaluation writes a distinct measured-result JSON, labeled `new_evaluation_of_unverified_checkpoint`. This never overwrites `results/reported_metrics.json`.

Evaluation has no random shuffle. It uses the model's original library metrics, including background and probability scores rather than argmax. It records batch size and sample count. A differently batched evaluation can produce different aggregate values, so exact historical comparisons require the original evaluation recipe. No independent test score or official xView2 score is implied.

Candidate TensorFlow checkpoints live outside this repository. Use the checkpoint **prefix** ending in `best_model.ckpt`, with both `.index` and `.data-00000-of-00001` alongside it. Do not pass `.pth`, the extracted PyTorch `best_model/` directory, or an individual TensorFlow shard. Model-variable restore compatibility does not identify epoch count.

A fresh training command begins a new run; it does not resume the historical 105-epoch run. The wrapper intentionally supplies no implicit resume or epoch-number repair. If recovery later identifies a resumable optimizer state, that procedure should be added as a new dated change.

## Direct raw-scene training, prediction and evaluation

See [the later local workflow](LOCAL_WORKFLOW.md) for executable scene manifests, figures and evaluation without the missing numeric arrays. `train-scenes` starts a fresh, unaugmented baseline only with `--execute`; no original model is resumed. Its loss and soft metrics mask pixels with unknown-label ID255 when the chosen manifest policy is ignore. Its training/validation scene IDs and image paths must be disjoint. Scene separation does not prove absence of geographic/event leakage.

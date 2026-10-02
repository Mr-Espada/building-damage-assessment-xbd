# Reproducibility and evidence status

| Item | Status |
| --- | --- |
| Final written description | Masters `Final Report.pdf`, 42 pages; SHA-256 `7f3852f5fa2f5897f1f4cc2de7c08a7064a20e06e1fc540d0753234c0466fdf3` |
| Report architecture | Recovered: Keras U-Net, ResNet101, 512×512 RGB, five-channel softmax |
| Final candidate builder | `archive/final_candidate/model.py`, copied without code edits |
| Inference candidate | `archive/final_candidate/model_testing.ipynb`, original code with outputs removed |
| TensorFlow weights | Two compatible local candidates, excluded from Git; candidate06 is a plausible final-checkpoint candidate |
| Historical local inference | Hidden notebook names candidate06 and preserves four prediction outputs; exact loaded-byte identity unverified |
| Exact 105-epoch history | Not recovered |
| Exact final training/validation arrays and split/augmentation manifests | Not recovered |
| Exact original environment | Not recovered; notebook kernels alone are insufficient |
| Report validation IoU/F1 | Reported, unchanged, not independently reproduced |
| Full benchmark reproduction | Pending missing provenance; no retraining performed |

The original folder mixes several projects. `Model_Tensorflow/Uncertain_logs/*.pth` are PyTorch ResUNet++ artifacts. Their 30-entry logs include NaN validation loss/F1 and do not verify the report. `unet_model_cvn.ipynb` is a separate ResNet50 four-class 224-pixel microscopy example. Later crop/SAM experiments are not substitutes for the final pipeline.

The report says LR=1e-4, planned epochs=150, used checkpoint=105 epochs and validation=7,464 tiles. The recovered builder defaults to 1e-5; `Uncertain_logs/train.ipy` uses 10 epochs and truncates validation to 1,866 array items. It imports a `build_model(weights)` signature that does not match the main builder. These discrepancies are retained visibly.

The recovered model metrics are `segmentation_models.metrics.iou_score` and `f1_score`. Under the cleanup library version, defaults are no probability threshold, averaging over all five classes including background, aggregation across batch pixels (`per_image=False`), and smoothing 1e-5. Averaging batch metric values is not the same as pooling an entire split, and macro IoU/F1 do not obey a single global IoU-to-F1 identity. The report's exact original package versions and aggregation details remain unverified. [Library source](https://github.com/qubvel/segmentation_models/blob/1.0.1/segmentation_models/metrics.py).

The cleanup commands are usable with numeric arrays supplied by the owner. They preserve the recovered model architecture, loss and metrics but improve paths, data validation and run logging. Seeded ordering and batching are later implementation choices. They cannot reconstruct missing data provenance or guarantee the original scores.

To close the evidence gap, recover the final crop/augmentation driver, numeric arrays or a scene/tile manifest, complete 105-epoch logs, original package versions and the exact used checkpoint. Associate qualitative figures with scene IDs, tile coordinates and that checkpoint. Verify those artifacts before claiming exact reproducibility. Any fresh preparation/training/evaluation phase must create a separate run record and preserve this report snapshot.


## Historical HPC training and checkpoint provenance

Runtime inspection matched both candidates to the 51,606,046-parameter U-Net/ResNet101. Candidate05 stores 180,606 optimizer iterations and candidate06 stores 600,078. With 186,404 samples, global batch 32, ceiling batching, a counter starting at zero and no reset/preceding updates, these are exactly 31 and 103 epoch equivalents. There is no explicit epoch metadata. Saved LR approximately 1e-5 is a checkpoint snapshot, not proof of initial LR; the report-versus-builder configuration remains unresolved.

The retained partial training recipe uses best-validation-loss checkpointing. A run ending at 105 could keep an earlier best state, so the conditional 103 equivalent is not by itself a contradiction. The recipe is not authenticated to the final run, and no history ties that scenario or the report's metrics to either checkpoint.

The hidden inference notebook supplies new positive local-testing evidence, but not an HPC identity or complete report-figure chain. Classification: **Plausible final-checkpoint candidate** for candidate06. See [the detailed evidence tiers, timeline and optimizer interpretation](HISTORICAL_HPC_PROVENANCE.md) and [checkpoint checksums](../results/checkpoint_candidates.json). Exact historical reproduction remains incomplete; an explicitly disclosed archival release does not require a new experiment.

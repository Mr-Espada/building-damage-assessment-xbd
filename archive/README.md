# Experiment catalogue

| Original experiment | Treatment |
| --- | --- |
| TensorFlow U-Net/ResNet101 builder and inference notebook | Selected final implementation candidates; original builder/helpers and stripped notebook in `final_candidate/`. |
| TensorFlow weighted builder / 10-epoch training fragment | Locally archived alternative; not the report's complete training run. |
| TensorFlow ResNet50 microscopy notebook | Unrelated template with unresolved attribution; local archive only. |
| PyTorch U-Net ResNet18 / DeepLabV3 / FPN / ResUNet++ | Earlier/alternative experiments; exact files preserved in local archive, with missing-module/syntax/custom-fork issues documented in the audit. |
| 224-pixel per-building crop experiments | Later 2025 work, distinct from final semantic segmentation. Local archive only. |
| SAM and SAM conversion helpers / SAM geotiff formats | Later 2025 experiments; training/gradient problems identified. Local archive only. |
| Statistics and plot notebooks | Report-related analysis mixed with later annotation statistics. Exact notebooks and outputs preserved locally, not bundled into this candidate. |
| Hidden notebook checkpoints | Exact duplicates deduplicated in selection; nonidentical variants preserved in the local archive. |

The original folder remains the authoritative raw archive. “Alternative” does not imply that the owner intentionally abandoned an experiment. Do not combine scores from these variants.

`final_candidate/` files retain original code and working-directory assumptions. They are for historical inspection; the supported explicit-path workflow is under `src/`. The inference notebook has a provenance notice and stripped outputs, but its code was not rewritten into a different experiment.

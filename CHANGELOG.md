# Changelog

## Original undergraduate work — 2023

- Final report describes xBD damage segmentation using U-Net/ResNet101, Dice loss and Adam; reported validation IoU 0.62743 and F1 0.6746 at 105 epochs.
- Recovered artifacts include a matching Keras builder, an inference notebook, two candidate TensorFlow checkpoints, alternative PyTorch models and report/statistics material. Exact original environment/run provenance is incomplete.

## Later experiments — recovered 2025 artifacts

- Building-level 224-pixel crops, memory maps, HDF5 and SAM conversion/fine-tuning attempts are present in the original folder. File timestamps and notebook metadata support this later grouping; their success and abandonment status are not assumed.

## Repository preparation — 2026-10-01

1. Inventoried every original file and directory; recorded logical sizes, source hashes, notebook metadata, exact source duplicates, errors, paths and scoped credential scan. No original file was edited, moved or deleted.
2. Used the Masters project's 42-page report as authority; identified the capstone folder's 24-page PDF as an older write-up. Preserved both originals locally, excluded full PDFs from the public candidate.
3. Identified `Model_Tensorflow/model.py` and `model_testing.ipynb` as final implementation candidates. Disclosed LR, epoch, validation and preprocessing differences rather than modifying reported claims.
4. Identified PyTorch ResUNet++ artifacts under `Uncertain_logs`; exported small histories into plain JSON with non-finite values represented explicitly. Kept their results separate from the final report.
5. Created a separate small release candidate and local exact-source archive. Excluded all dataset trees, checkpoints, serialized arrays, memory maps, CSV/HTML outputs, bytecode and unrelated templates from the release candidate. No huge-file deletion or migration occurred.
6. Preserved byte-identical final candidate builder/helper sources in `archive/final_candidate/`; stripped outputs, execution counts and nonessential notebook metadata from the public inference notebook copy. Kept its code unchanged. Exact original notebooks and hidden variants remain in the local archive.
7. Added `src/building_damage/` with explicit-path array validation and train/evaluate wrappers. Set legacy Keras before imports, made ImageNet initialization explicit, retained original `/255` normalization and library loss/metrics, and added report-versus-builder configs. New wrappers are labeled later work. Their seeded ordering, validation checks and streaming batches are not represented as original run behavior.
8. Added README, environment/dependency instructions, dataset preparation/storage instructions, workflow docs, limitations, experiment catalogue, references, license recommendation, NOTICE, publication checklist and .gitignore.
9. Transcribed report metrics unchanged to JSON and plotted a labeled summary. No predictions, new model results, benchmark improvements or confidence intervals were invented.
10. Ran package/dependency, synthetic-data, source-preservation, notebook, documentation and release-content checks. Inspected/restored candidate weights without training or evaluating original imagery. Detailed scope and outcomes are in the local audit's `VALIDATION.md`.

## Second-pass historical provenance review — 2026-10-01

1. Expanded the read-only source/path review to 108 ordinary source/config/text/state files, 34 notebooks and static strings from 32 bytecode files; searched for HPC/scheduler, environment, transfer and final-run artifacts. No originals were changed or executed.
2. Compared visible and hidden inference notebooks; documented identical source and the hidden notebook's sequential predictions/four PNG outputs. Refined the first audit's inference evidence without treating saved output as an authenticated checkpoint digest.
3. Compared both checkpoint tensor states and object graphs. Recorded distinct model/optimizer tensors, conditional epoch arithmetic, optimizer continuity limits and saved-LR-versus-initial-LR distinction. Documented how best-only checkpointing could reconcile a 105-epoch run with an earlier saved state, without asserting it occurred.
4. Reviewed report Figures 13–17 against 26 saved notebook PNGs, 11 graphs and five standalone illustrations. Documented partial site/actual-mask links and the absent complete prediction chain. Kept all comparison images private and did not produce a model metric.
5. Added `docs/HISTORICAL_HPC_PROVENANCE.md` and `docs/APPLICATION_WORDING.md`; updated README, reproducibility/results notes, checkpoint metadata and publication checklist. Classified candidate06 as a plausible final-checkpoint candidate, distinguishing artifact evidence, author recollection and inference.
6. Corrected the release author's full name to **Mohammad Hachim Eraissouni** in README, NOTICE and the capstone bibliography entry, verified against the report cover.
7. Preserved first-pass prepared-file versions in the private audit's `revisions/first-pass/`, added forensic evidence/validation and a change manifest, and rebuilt the candidate ZIP. No historical source, reported metric/chart, dataset, weight, training wrapper or dependency recipe changed. No training or real-data evaluation was performed.

## Licensing and local Git preparation — 2026-10-01

1. Recorded the author's personal code-ownership confirmation and delegated license choice. Added the standard MIT LICENSE with copyright 2023–2026 Mohammad Hachim Eraissouni; kept dependencies and excluded data/imagery/report material under their separate terms.
2. Updated README, NOTICE, license guidance, publication checklist and provenance publication notes to reflect the selected license rather than a pending recommendation.
3. Added full author, README, SPDX MIT expression and license-file metadata to `pyproject.toml`. Raised the minimum build-tool requirement to Setuptools 77 for SPDX/license-files support; the validated lock already uses 84.0.0. Runtime dependencies are unchanged.
4. Updated the release checker to allow the root LICENSE explicitly while retaining its file-type, secret-pattern, size, notebook-output and image restrictions.
5. Preserved pre-license prepared versions privately, rebuilt the release ZIP and checked its exact public file contents. Validated package license metadata and included notices without model execution or dependency installation.
6. Initialized local Git on `main` only in the prepared public-release directory, staged the reviewed files and checked their names/hashes against the release manifest. No initial commit, remote or publication was created. Historical sources, checkpoint data, reported metrics and scientific findings are unchanged.

## Initial Git commit preparation — 2026-10-02

1. Verified the 37-file public selection against the staged Git index, license scope, notebook sanitation and exclusions before publication. Reported results remain unchanged and not independently reproduced.
2. Preserved prepared versions before the first commit privately and prepared the initial `main` commit. Scientific author attribution and copyright remain unchanged; requested shared repository management concerns GitHub permissions.
3. Separated local commit preparation from subsequent GitHub publication and access verification. Personal repositories cannot provide a second owner; shared administration requires an organization repository. The user selected publication under personal account `Mr-Espada`, with `mehdinejjar86` invited as a write collaborator; this preparation entry does not assert the invitation was accepted.

## Public GitHub publication — 2026-10-02

1. Published the reviewed research archive under `Mr-Espada/building-damage-assessment-xbd` with MIT licensing and verified anonymous public access. The repository contains the selected code/documentation and reported-value chart; original datasets, checkpoints, full PDFs and private audit/archive material are excluded.
2. Issued the requested write-collaborator invitation. Activation depends on acceptance; management access does not change academic authorship or copyright attribution.
3. Updated README status, publication checklist, cleanup provenance and local navigation/audit records; rebuilt the release ZIP and verified local/remote file identity. No scientific method, reported metric, dataset or checkpoint changed.

## Local scene workflow preparation — 2026-10-02

1. Added `src/building_damage/scenes.py` for deterministic, metadata-only manifests of paired post-disaster TIFF/JSON scenes and streamed, range-checked loading. It records content hashes, keeps all four 512-pixel tiles, performs no resampling or augmentation, and requires an explicit unknown-subtype policy when the default rejection encounters one. This is a fresh preparation protocol, not recovery of the historical final arrays.
2. Added `src/building_damage/scene_workflow.py` and CLI dispatch for `prepare-scenes`, `predict-scenes`, `evaluate-scenes` and `train-scenes`. Restored-checkpoint prediction/evaluation uses local weights without ImageNet initialization; prediction saves class arrays and labeled input/mask/prediction/overlay figures. Manifests, source/checkpoint hashes, configuration, runtime versions, progress and completion/failure status distinguish new runs from reported results. Existing output paths are refused.
3. Added `src/building_damage/metrics.py` for dataset-level argmax confusion counts, per-class scores, macro scores with documented undefined-class handling, pixel accuracy and ignored-pixel counts. Evaluation keeps these separate from the recovered probability-based library metrics, records effective batching and labels every new result as distinct from historical report reproduction and the official xView2 score.
4. Added `configs/local_inference.json` with no encoder initialization and batch size 1, plus `tests/test_scenes.py`, `tests/test_metrics.py` and `tests/test_scene_workflow.py` for geometry/data contracts, hand-calculated score behavior and run contracts. These additions do not claim validation of the original 105-epoch experiment.
5. Added `docs/LOCAL_WORKFLOW.md` and updated public workflow/environment/data/reproducibility guidance and README references. Personal path profiles, manifests, generated results and checkpoint files remain local and excluded from Git.
6. Prepared a future fresh-training interface: scene-level split overlap checks, matching unknown policies, no augmentation and no implicit checkpoint resume. The recommended future plan uses tier1+tier3 for training, hold for validation and reserves test. It differs explicitly from the report's preparation and split usage. No retraining is implied by workflow preparation, and report metrics/chart remain unchanged.


7. Fixed the cleanup wrapper’s seeded-initializer compatibility with Python3.12/tf-keras2.16 using Python/NumPy/TensorFlow seed setters; added a regression check. No historical code or checkpoint tensor changed.
8. Recorded runner source/config hashes and device inventory; marked evaluation progress completed/failed and excluded entirely ignored batches from probability-score means.

9. Validated 24 automated checks and the persistent local environment; restored retained candidate06 and generated a fresh four-tile post-disaster scene figure. Verified complete hold+test (1,866 scenes/7,464 tiles) and fresh training (9,168 scenes/36,672 unaugmented tiles; hold validation) plans without launching full evaluation or training. Checked soft aggregation against legacy Keras on synthetic probabilities. No new benchmark score or reported-metric change was made.

## Private storage organization — 2026-10-02

1. Consolidated 172 original sources/notebooks and historical evidence files into a private TAR archive; reopened and SHA256-verified every member. Preserved the authoritative report separately without modifying synced references.
2. Moved all44,136 raw pre/post image/annotation files and both complete TensorFlow checkpoint sets to canonical private storage on the same filesystem. Verified raw file inode/size/mtime and checkpoint hashes; updated the owner's private path profile.
3. Preserved allnine unique alternative PyTorch/SAM checkpoint files in private history. Verified that the extracted PyTorch tensor payload duplicates a retained checkpoint byte for byte.
4. Verified a fresh post-disaster scene prediction after migration: source hashes, selected checkpoint hashes and predicted class arrays match the earlier local run. No training, full evaluation or report-metric change occurred.
5. Moved remaining legacy material into a reversible private holding area, including retired224crop/SAM derivatives and archived working copies. Permanent deletion of the exact set is pending explicit approval; no deleted-data or reclaimed-storage claim is made at this phase.
6. Updated private navigation, the local guide and generic storage documentation. Historical archive contents and earlier run manifests retain their original bytes/paths; a private migration manifest maps the old layout to current locations. No weights, raw data, private archive or newly generated imagery were published.

Future corrections, recovered provenance and new experiments must receive distinct dated entries. Recovered logs should augment the record; they must not silently replace the original report values.

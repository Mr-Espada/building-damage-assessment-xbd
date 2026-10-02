# Dataset acquisition, storage and preparation

xBD provides before/after satellite images and building annotations for disaster damage assessment. Acquire it through [xView2](https://xview2.org/dataset). The [CMU SEI project page](https://www.sei.cmu.edu/projects/xview-2-challenge/) identifies a Creative Commons license. The exact current dataset/imagery terms could not be verified through the text-only xView2 page on 1 October 2026; read the site's terms and retain a dated copy when acquiring data. No automatic downloader or third-party mirror is supplied.

Keep originals immutable in external storage. A future arrangement can be:

```text
external_storage/
  raw/xbd/{tier1,tier3,hold,test}/{images,labels}/
  derived/final_tiles/       # numeric arrays + scene/tile/augmentation manifests
  derived/building_crops/    # later 224-pixel experiment; separate provenance
  derived/sam/               # later SAM formats; separate provenance
  checkpoints/tensorflow_candidates/
  checkpoints/alternatives/
```

The existing original folder contains roughly 336 GB across raw/derived data, memory maps and weights. Leave it in place during cleanup. Pass existing paths explicitly; do not duplicate it into this repository, put external data symlinks inside it, or initialize Git at the original folder's root.

## Report-described final preparation

1. Select **post-disaster** TIFF images and their matching xBD JSON pixel-coordinate (`features.xy`) annotations. Training: tier1+tier3, 9,168 source scenes. Validation: hold+test, 1,866 scenes. Preserve scene IDs and split membership before generating variants.
2. Rasterize damage masks with pixel classes 0 background, 1 undamaged, 2 minor, 3 major, 4 destroyed. The legacy helper uses integer-cast polygon coordinates and maps every unrecognized/missing subtype to class 1. Record this as legacy behavior; a corrected unknown-class policy is a later methodological change.
3. The report describes training augmentation by rotation, flipping and sharpening for scenes with minor/major/destroyed annotations, with duplicate removal. It describes four 512×512 tiles per 1024×1024 scene and removal of empty tiles within annotated training scenes. Validation is not augmented; 1,866×4 gives 7,464 validation tiles.
4. Save RGB image arrays `(N,512,512,3)` and integer mask arrays `(N,512,512)` with dtype `uint8`; normalization to float32 `/255` and one-hot encoding occur at batch loading time. Record input archive checksums, preparation version, scene ID, tile row/column, augmentation, class mapping and output checksums in a manifest.

**Recovery gap:** the exact final crop/augmentation driver and arrays are absent. The report's 186,404 training tiles cannot currently be regenerated with established provenance. The existing `save_data.py` resizes images instead of applying the final tiling recipe and interpolates categorical masks linearly. `Preparing_data.py` makes 224-pixel building crops; `data/X.npy` and `data/y.npy` have object dtype and incompatible dimensions. None is a verified drop-in final preparation command.

The unchanged mask/cut helpers are retained in `archive/final_candidate/data_processing.py` for inspection. Do not run archived top-level preparation scripts against originals: some overwrite memory maps/HDF5 and use unsorted file listings. Preparation instructions document the recoverable recipe; no unvalidated end-to-end reconstruction is presented as the original pipeline.

For exact historical reconstruction, recover or implement a separately versioned driver, avoid interpolation of categorical masks, specify unknown labels and empty-tile rules, fix duplicate-index handling, split before augmentation, and validate on synthetic polygons before using data. Report any difference from the undergraduate method. Do not compare reconstructed scores as if they were the historical run.

## Later raw-scene workflow — October 2026

The [local workflow](LOCAL_WORKFLOW.md) supplies a separately labeled, executable raw-scene path. It indexes image/annotation pairs without reading image payloads, then validates/rasterizes one scene at a time during prediction, evaluation or fresh training. It retains every tile, performs no augmentation or resizing, validates integer RGB values before uint8 conversion (the recovered TIFFs use int16), and exposes unknown-label error/ignore/legacy-undamaged policies. This does not regenerate the historical augmented training arrays or settle the report's ambiguous empty-tile filtering prose.

# Environment

`requirements.txt` pins direct dependencies for a **cleanup compatibility environment**, with a full Linux/Python 3.12 dependency snapshot in `requirements-lock.txt`. This is not a recovered 2023 environment. Historical notebooks identify Python 3.8/3.9 for final-candidate work; later alternatives use 3.10/3.13. They provide no complete package inventory.

Use Python 3.12, TensorFlow 2.16.2, `tf-keras` 2.16.0, Segmentation Models 1.0.1 and NumPy 1.26.4 in an isolated environment. The package sets `TF_USE_LEGACY_KERAS=1` and `SM_FRAMEWORK=tf.keras` **before** importing TensorFlow/Segmentation Models. Set these variables yourself before imports in external notebooks. A fresh process is needed if Keras 3 was already initialized.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

The lock snapshot is for the validated Linux/Python 3.12 CPU inspection environment. It is a version snapshot without wheel hashes, not a cross-platform GPU guarantee. Use direct requirements if platform-specific wheels require resolution, then record the resulting versions. Tests do not train or use original data. Current GPU driver/CUDA compatibility must be checked for a future training machine; original report hardware was four A40 GPUs and substantial RAM. [TensorFlow installation guidance](https://www.tensorflow.org/install/pip).

Inference/evaluation builds with `encoder_weights=None` before restoring a checkpoint, avoiding redundant ImageNet downloads. New training explicitly selects ImageNet encoder weights and may download them. The original helper inherited this library default. Legacy `/255` input scaling is preserved; switching to a different backbone preprocessing function would be a methodological change.

Optional Jupyter tooling is separate: install `jupyterlab` in your chosen notebook environment and select this environment as kernel. Alternative PyTorch/SAM code is not supported by the final environment. Its missing modules/custom ResUNet++ fork must be recovered before creating a separate environment; a speculative PyTorch dependency file is not supplied.

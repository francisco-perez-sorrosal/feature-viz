# Sample images

A curated set of 196 images (one per varied ImageNet class) backing the
dataset-examples depiction in `notebooks/01_neurons_and_features.py`.

- **Source**: the [`imagenet-sample-images`](https://github.com/EliSchwartz/imagenet-sample-images)
  repository — one representative JPEG per ImageNet-1k class.
- **Provenance / regeneration**: `scripts/fetch_sample_images.py` (the class
  list and download/resize logic). Re-run it to refresh the set.
- **Licensing**: these are ImageNet sample images, bundled for **non-commercial
  educational use only**. Do not redistribute for other purposes.

They are downscaled to a 384 px longest edge to keep the bundle small (~5.5 MB).

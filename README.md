# feature-viz

An educational, POC-grade reimplementation of three ideas from Chris Olah et
al.'s *Circuits* thread on Distill (2020). It makes concrete the three questions
I hit when reading those articles:

1. How is a "neuron" actually represented inside InceptionV1? — [*Zoom In*](https://distill.pub/2020/circuits/zoom-in/)
2. How can we visualize the weights that connect two neurons? — [*Visualizing Weights*](https://distill.pub/2020/circuits/visualizing-weights/)
3. How is the small synthesised RGB thumbnail representing each neuron actually produced? — [*Early Vision*](https://distill.pub/2020/circuits/early-vision/)

Each question gets a [marimo](https://marimo.io) notebook that explores it
interactively, backed by a small reusable library (`feature_viz`). A fourth
notebook composes notebooks 01 and 02 into a **circuit** view — Claim 2 of
*Zoom In* made concrete.

> [!NOTE]
> Model: `torchvision.models.googlenet` with `IMAGENET1K_V1` weights — a
> faithful re-implementation of InceptionV1. Every structural and algorithmic
> claim from the papers carries over, but the channel indices do **not** match
> OpenAI Microscope's TensorFlow checkpoint.

## Setup

Requires [uv](https://docs.astral.sh/uv/). The first `uv` command creates an
isolated `.venv` and installs everything, including the `feature_viz` package
itself (editable):

```bash
cd ~/dev/feature-viz
uv sync
```

## Running the notebooks

```bash
uv run marimo edit notebooks/01_neurons_and_features.py    # interactive editor
uv run marimo run  notebooks/03_feature_visualization.py   # read-only app view
```

The four notebooks:

| Notebook | Question | What it shows |
| -------- | -------- | ------------- |
| `01_neurons_and_features.py` | What *is* a neuron, and what does it detect? | Browse channels with the **scout**, then inspect one neuron three ways at once: its **activation map** (raw `H×W` numbers), its **response** (the same map on your image), and its **feature** (receptive-field crops of the dataset images it fires on), bridged by an activation ranking. |
| `02_weights_between_neurons.py` | What connects two neurons? | The effective `k×k` kernel `W_eff[j,i]` — BatchNorm folded in, bottleneck multiplied out. Red excites, blue inhibits. |
| `03_feature_visualization.py` | How is the DxD thumbnail made? | Activation-maximisation gradient ascent that synthesises the input image driving a channel hardest, with a step-by-step evolution filmstrip from noise to feature. |
| `04_circuits.py` | What is a *circuit*? | The composition of 01 + 02: pick a downstream neuron, then the top excitatory and inhibitory upstream channels are ranked and shown as `[weight matrix] + [upstream feature crops]` next to the downstream feature — *these features, weighted this way, build that one.* |

marimo notebooks are plain `.py` files — they diff cleanly in git and are edited
like any source file. Cells are reactive: change a control and dependents
recompute. Notebook 03's render is gated behind a button so the ~18 s
optimisation only runs on demand.

## Using the library directly

The notebooks are thin visual wrappers; all logic lives in `src/feature_viz/`
and is usable on its own:

```python
import feature_viz as fv

model = fv.load_inception(fv.best_device())
result = fv.render_neuron(model, "inception4b", channel=100, steps=512, n_snapshots=12)
# result.image -> uint8 [H,W,3];  result.activations -> the ascent curve
# result.snapshots -> [(step, image), ...] the optimisation trajectory
```

| Module | Responsibility |
| ------ | -------------- |
| `device` | Runtime capability probe → picks CPU / CUDA / MPS. |
| `model` | Loads InceptionV1; named-layer lookup; activation-capture hook. |
| `neuron` | Activation capture and channel-slice statistics (notebook 01). |
| `dataset_examples` | Runs the bundled `sample_images/` through the model; produces receptive-field crops at each neuron's firing peak — top-K per channel and one card per channel for the scout (notebook 01). |
| `weights` | BatchNorm folding and the effective `k×k` kernel (notebook 02). |
| `feature_vis` | `FourierImage`, transformation robustness, the render loop (notebook 03). |
| `plotting` | matplotlib helpers the notebooks render. |

`src/feature_viz/sample_images/` holds ~196 bundled ImageNet sample images for
the notebook-01 dataset-example view; `scripts/fetch_sample_images.py`
regenerates them (see that directory's `README.md` for provenance and licensing).

## Device note (Apple Silicon)

On an M2, `best_device()` selects **CPU** for the feature-visualisation render,
and this is deliberate, not a fallback to "safe mode":

- The Fourier-basis FFT (`irfft2`) runs fine on MPS.
- The rotation in the transformation-robustness step uses `grid_sample`, whose
  **backward pass is unimplemented on the MPS backend** (PyTorch 2.12). With
  `PYTORCH_ENABLE_MPS_FALLBACK=1` it round-trips to CPU every step — measured
  *slower* than plain CPU (112 vs 70 ms/step).
- Even with rotation removed, MPS only beats CPU ~1.14× for this small model.

`device.probe_mps_render()` tests the real op set at runtime. If a future
PyTorch implements the missing op, MPS is selected automatically — no code
change. Forward-only work (notebooks 01–02) is instant on CPU regardless.

## What this skips

Deliberate scope — these are simplifications, not bugs to fix:

- BatchNorm folding in the render (notebook 03 uses the model as-is; only the
  weight analysis in notebook 02 folds BN explicitly).
- The ReLU between the `1×1` and `k×k` convs when multiplying out `W_eff` — the
  same approximation the Distill paper makes.
- Diversity terms, per-neuron objective variants, and frequency-band curriculum
  from the *Feature Visualization* paper.

For production-grade visualisations use [`lucent`](https://github.com/greentfrapp/lucent)
(PyTorch port of lucid) or the original [`lucid`](https://github.com/tensorflow/lucid).

## References

- Olah et al., [*Zoom In: An Introduction to Circuits*](https://distill.pub/2020/circuits/zoom-in/), Distill, 2020
- Olah et al., [*An Overview of Early Vision in InceptionV1*](https://distill.pub/2020/circuits/early-vision/), Distill, 2020
- Voss et al., [*Visualizing Weights*](https://distill.pub/2020/circuits/visualizing-weights/), Distill, 2020
- Olah, Mordvintsev & Schubert, [*Feature Visualization*](https://distill.pub/2017/feature-visualization/), Distill, 2017
- [OpenAI Microscope — InceptionV1](https://microscope.openai.com/models/inceptionv1/)

import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(r"""
    # 3 — Feature visualisation: a neuron → the image it wants

    Based on Distill, [*Early Vision in InceptionV1*](https://distill.pub/2020/circuits/early-vision/)
    and the method paper [*Feature Visualization*](https://distill.pub/2017/feature-visualization/).

    ## The one idea

    Notebook 01 found *what a neuron detects* by searching a dataset for
    images that activate it. This notebook does the opposite: instead of
    *searching* for an image, it **synthesises one from scratch**.

    Start from random noise, then repeatedly nudge the **pixels** — never the
    network — to make one chosen neuron fire harder. After a few hundred
    steps the image becomes a crisp, idealised picture of that neuron's
    feature: a curve detector grows curves, a fur detector grows fur.

    ## Why it matters — evidence for Claim 1

    Claim 1 says a feature can be *understood*. But how do you *prove* a neuron
    is, say, a curve detector? You gather **independent lines of evidence** that
    agree:

    - **Dataset examples** (notebook 01) — real images it fires on.
    - **Feature visualisation** (this notebook) — a synthetic image built only
      from the neuron's own gradient, touching no dataset.

    The two methods share no inputs. When both independently show curves, the
    label "curve detector" is hard to argue with. That agreement *is* what the
    paper means by understanding a feature.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        graph LR
            P["Fourier parameters<br/>(the only learned thing)"] --> IMG["image x<br/>3 x 224 x 224"]
            IMG --> T["random transform<br/>jitter / scale / rotate"]
            T --> NET["frozen InceptionV1"]
            NET --> A["target channel<br/>activation"]
            A --> OBJ["objective<br/>= - activation"]
            OBJ -->|"gradient ascent updates the parameters"| P
        """
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("**Concepts** — expand any term you want defined:"),
            mo.accordion(
                {
                    "Feature visualisation": mo.md(
                        "Synthesising an input image that maximally activates a "
                        "chosen neuron, so you can *see* what it detects — without "
                        "any dataset."
                    ),
                    "Activation maximisation / gradient ascent": mo.md(
                        "Training normally does *gradient descent* on the weights "
                        "to lower a loss. Here we do *gradient ascent* on the "
                        "**image** to raise a neuron's activation. Same machinery, "
                        "different variable."
                    ),
                    "Frozen network": mo.md(
                        "The model's weights are fixed (`requires_grad = False`). "
                        "Gradients flow only into the image parameters — the "
                        "network is the unchanging measuring instrument."
                    ),
                    "Fourier parameterisation": mo.md(
                        "The image is stored not as pixels but as **frequency "
                        "coefficients** (a 2-D Fourier basis). Optimising in this "
                        "space is a natural-image prior — it steers away from "
                        "pixel-level adversarial noise."
                    ),
                    "1/f prior": mo.md(
                        "Natural images have more energy in low frequencies than "
                        "high. Scaling the spectrum by `1/frequency` bakes that in, "
                        "suppressing the high-frequency static a raw optimiser loves."
                    ),
                    "Transformation robustness": mo.md(
                        "Each step the image is randomly jittered, scaled and "
                        "rotated before the forward pass. The result must then "
                        "activate the neuron under *all* those views — so it "
                        "becomes a robust feature, not a fragile single-pixel trick."
                    ),
                    "Receptive field": mo.md(
                        "The input region a neuron can see. Deep-layer neurons "
                        "have large receptive fields, so their synthesised images "
                        "look like whole objects; early layers give textures."
                    ),
                    "Union of evidence": mo.md(
                        "No single method *proves* what a neuron detects. The "
                        "paper's standard is **agreement between independent "
                        "methods** — dataset examples (notebook 01) and feature "
                        "visualisation (here). If both show the same motif, the "
                        "feature is understood."
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                "**Background** — expand if neural-network optimisation is new to you:"
            ),
            mo.accordion(
                {
                    "Gradient": mo.md(
                        "For a chosen number (here, a neuron's activation), the "
                        "**gradient** says how to change each input to increase it "
                        "fastest. Automatic differentiation computes it for free."
                    ),
                    "Loss / objective": mo.md(
                        "The single number an optimiser steers. Training minimises "
                        "a *loss*; here the objective is `-activation`, so "
                        "minimising it **maximises** the neuron's activation."
                    ),
                    "Optimiser (Adam)": mo.md(
                        "The rule that turns a gradient into an actual update step. "
                        "**Adam** is a robust default — it adapts the step size per "
                        "parameter. Here it updates the image, not the network."
                    ),
                    "Pixels vs. frequencies": mo.md(
                        "An image can be described by its **pixels**, or "
                        "equivalently by **frequencies** (how much coarse vs. fine "
                        "detail it has). This notebook optimises in frequency space "
                        "— see *Fourier parameterisation* above."
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — the recipe (the canonical 'lucid' method)

    Four ingredients turn random noise into a feature picture. Expand each in
    the concepts above; in short:

    1. **(a) Fourier parameterisation + 1/f prior** — the image lives as
       frequency coefficients, decorrelated and 1/f-scaled. This is a
       natural-image prior: it blocks adversarial high-frequency noise.
    2. **(b) Colour decorrelation + sigmoid** — channel values are mixed
       toward real ImageNet colour statistics, then a sigmoid keeps RGB in
       `(0, 1)`.
    3. **(c) Transformation robustness** — random jitter / scale / rotate
       every step, so the feature must survive small perturbations.
    4. **(d) Gradient ascent** — forward through the *frozen* network, read
       the target channel's mean activation, and ascend on the image
       parameters.

    Loop (d) a few hundred times and the image converges to the neuron's
    idealised input.
    """)
    return


@app.cell
def _():
    import feature_viz as fv
    from feature_viz.plotting import show_ascent_curve, show_image, show_image_grid

    return fv, show_ascent_curve, show_image, show_image_grid


@app.cell
def _(fv):
    # Frozen model — gradient ascent optimises the image, never the weights.
    device = fv.best_device()
    model = fv.load_inception(device)
    return device, model


@app.cell
def _(device, mo):
    _on_mps = device.type == "mps"
    mo.callout(
        mo.md(
            f"**Render device: `{device}`.** "
            + (
                "MPS — the runtime render-path probe passed."
                if _on_mps
                else "CPU was chosen on purpose. The rotation in step (c) uses "
                "`grid_sample`, whose backward pass is unimplemented on Apple's "
                "MPS backend; with a CPU fallback it is *slower* than plain CPU "
                "(measured 112 vs 70 ms/step). `device.probe_mps_render()` checks "
                "this at runtime, so a future PyTorch that adds the op switches "
                "to MPS automatically."
            )
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2 — choose a neuron and render

    Pick a layer and a channel. As in notebook 01, channel numbers do **not**
    match the Distill articles — deeper layers (`inception4*`, `inception5*`)
    give more object-like images; early layers give textures and patterns.

    The render is the expensive step (~10–20 s), so it is gated behind a
    button — marimo would otherwise re-run it on every control change.
    """)
    return


@app.cell
def _(fv, mo, model):
    layer = mo.ui.dropdown(
        options=fv.inception_blocks(model), value="inception4b", label="Layer"
    )
    layer
    return (layer,)


@app.cell
def _(fv, layer, model):
    # Channel count is layer-specific — derive it instead of hard-coding.
    n_channels = fv.layer_channels(model, layer.value)
    return (n_channels,)


@app.cell
def _(mo, n_channels):
    channel = mo.ui.number(
        start=0, stop=n_channels - 1, value=100, label=f"Channel (0–{n_channels - 1})"
    )
    steps = mo.ui.slider(
        start=64, stop=768, step=64, value=256, label="Ascent steps", show_value=True
    )
    run_button = mo.ui.run_button(label="Render image")
    mo.vstack([channel, steps, run_button])
    return channel, run_button, steps


@app.cell
def _(channel, fv, layer, mo, model, run_button, steps):
    mo.stop(
        not run_button.value,
        mo.md("Set the controls above, then click **Render image**."),
    )
    result = fv.render_neuron(
        model,
        layer.value,
        int(channel.value),
        steps=int(steps.value),
        seed=0,
        n_snapshots=12,
    )
    return (result,)


@app.cell
def _(channel, layer, mo, result, show_ascent_curve, show_image):
    mo.vstack(
        [
            mo.md(f"### Synthesised feature for `{layer.value}:{channel.value}`"),
            mo.hstack(
                [
                    show_image(result.image, title="the image the neuron 'wants'"),
                    show_ascent_curve(result.activations),
                ],
                justify="start",
                gap=2,
            ),
            mo.md(
                f"**Left** — the synthesised image: random noise reshaped, over "
                f"{len(result.activations)} steps, into whatever drives this "
                f"neuron hardest. **Right** — the activation climbing each step. "
                f"A healthy run *rises then plateaus* (it has converged); final "
                f"activation **`{result.final_activation:+.3f}`**. A flat or noisy "
                f"curve means the neuron is weak here, or needs more steps."
            ),
        ]
    )
    return


@app.cell
def _(mo, result, show_image_grid):
    mo.vstack(
        [
            mo.md(
                "### How the image evolved\n\n"
                "The optimisation trajectory, sampled at 12 steps: **step 0 is the "
                "raw random noise** the run starts from; each later frame is the "
                "image at that step as gradient ascent sculpts it into the "
                "feature. Watch structure emerge low-frequency first — a "
                "consequence of the 1/f Fourier prior."
            ),
            show_image_grid(
                [(f"step {s}", img) for s, img in result.snapshots], ncols=6
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaway — Claim 1, confirmed

    - Feature visualisation **synthesises** a neuron's ideal input by
      gradient ascent on the *image*, with the network frozen.
    - The natural-image prior (Fourier + 1/f) and transformation robustness
      are what make the result a real feature instead of adversarial noise.
    - It is the **second, independent** way to see a feature. Dataset examples
      (notebook 01) and feature visualisation agreeing is the *union of
      evidence* that lets the paper claim a feature is genuinely understood.

    Across the three notebooks you have now seen **Claim 1** in full — features,
    via two independent methods — and the start of **Claim 2** — the weights
    that wire features into circuits (notebook 02). Claim 3, universality, is
    left to the Distill articles.

    For production-grade visualisations use
    [`lucent`](https://github.com/greentfrapp/lucent) — this repo stays
    minimal on purpose.
    """)
    return


if __name__ == "__main__":
    app.run()

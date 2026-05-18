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
    # 2 — The weights *between* neurons

    Based on Distill, [*Visualizing Weights*](https://distill.pub/2020/circuits/visualizing-weights/).

    ## Where this sits — Claim 2

    Notebook 01 was **Claim 1**: features are the fundamental unit. This
    notebook opens **Claim 2 — Circuits**: features do not just sit there, they
    are **wired together by weights**. A *circuit* is a few features connected
    by weights that together compute something — and the wiring is just the
    conv weights, made visible here.

    ## The one idea

    Given an **upstream** neuron `i` (in an earlier layer) and a **downstream**
    neuron `j` (in a later one), what is the *weight* connecting them?

    In a fully-connected net a weight is one scalar. In a CNN it is **not** — it
    is a small **`k × k` spatial matrix** `W[j, i, :, :]`. Each entry says: *if
    upstream neuron `i` fires at this relative position in the receptive field,
    how much does it push downstream neuron `j` up or down?*

    Positive entries **excite** `j`; negative entries **inhibit** it. That
    matrix — not a number — is the connection.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **One honest approximation.** An Inception branch is two convs in
            series (`1×1` then `k×k`) with a **ReLU** between them. To get a
            *single* effective `k×k` matrix we multiply the two convs out and
            **ignore that ReLU** — i.e. treat the branch as linear. This is the
            same simplification the Distill article makes; it makes the weights
            visualisable, at the cost of being exact only where the ReLU is not
            clipping.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("**Concepts** — expand any term you want defined:"),
            mo.accordion(
                {
                    "Convolution kernel": mo.md(
                        "The small grid of weights a conv layer slides across its "
                        "input. Its tensor shape is `[C_out, C_in, k, k]` — a "
                        "`k x k` matrix for every (input channel, output channel) "
                        "pair."
                    ),
                    "Receptive field": mo.md(
                        "The patch of input that influences one output position. "
                        "A `k x k` weight matrix is read *over* the receptive "
                        "field: each cell is one relative spatial offset within it."
                    ),
                    "BatchNorm": mo.md(
                        "A per-channel normalise-scale-shift applied right after a "
                        "conv. At inference it is a *fixed* affine map, so it can "
                        "be **folded** into the conv weight: `W' = W · γ / √(var+ε)`. "
                        "We fold it so the weights shown are the ones truly applied."
                    ),
                    "Bottleneck": mo.md(
                        "A `1x1` conv placed before an expensive `k x k` conv to "
                        "shrink the channel count first — far cheaper compute. "
                        "Inception's `branch2` is exactly this: `1x1` then `3x3`."
                    ),
                    "ReLU": mo.md(
                        "The nonlinearity `max(0, x)` between the two convs. "
                        "Ignoring it (see the warning above) is what lets two "
                        "convs collapse into one linear `k x k` weight."
                    ),
                    "Excitatory / inhibitory": mo.md(
                        "A **positive** weight entry pushes the downstream neuron's "
                        "activation **up** (excites); a **negative** one pushes it "
                        "**down** (inhibits). Zero ≈ no connection."
                    ),
                    "Circuit": mo.md(
                        "A small group of **features connected by weights** that "
                        "together compute something — e.g. a curve detector built "
                        "from edge detectors at the right offsets. Claim 2 of the "
                        "paper: circuits, like features, can be understood."
                    ),
                    "Feature (recap from notebook 01)": mo.md(
                        "*What* a neuron detects — its identity (curve, edge, "
                        "fur…), seen via dataset crops (notebook 01) or feature "
                        "visualisation (notebook 03). A weight connects two "
                        "features; that is what makes it a *circuit*."
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
                "**Background** — expand if neural-network terms are new "
                "(notebook 01 has a fuller set):"
            ),
            mo.accordion(
                {
                    "Weight": mo.md(
                        "One of the network's learned numbers. A conv layer has "
                        "thousands; grouped into `k x k` filters they *are* what "
                        "this notebook visualises."
                    ),
                    "Training": mo.md(
                        "The process that *sets* the weights — repeatedly nudging "
                        "them so the network classifies ImageNet well. It is "
                        "already done; here we only read the finished weights."
                    ),
                    "Forward pass": mo.md(
                        "Running an image through the network to get activations. "
                        "Notebook 02 needs **no** forward pass at all — weights are "
                        "a fixed property of the layers, read off directly."
                    ),
                    "Upstream / downstream": mo.md(
                        "**Upstream** = closer to the input (earlier layer); "
                        "**downstream** = closer to the output (later layer). "
                        "Information — and these weights — flow upstream → "
                        "downstream."
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _():
    import feature_viz as fv
    from feature_viz.plotting import show_weight_grid, show_weight_matrix

    return fv, show_weight_grid, show_weight_matrix


@app.cell
def _(fv):
    # No forward pass needed — weights are read straight off the conv layers.
    model = fv.load_inception("cpu")
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — pick an Inception block

    Each Inception block runs four parallel **branches** and concatenates
    them. We look at **`branch2`**, a *bottleneck*: the `1×1` conv cheaply
    *reduces* the channel count, then the `3×3` conv does the spatial work.
    """)
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        graph LR
            UP["upstream channels<br/>C_in"] --> R["1x1 conv<br/>reduce"]
            R --> B(["bottleneck<br/>C_b"])
            B --> K["3x3 conv<br/>spatial mixing"]
            K --> DOWN["downstream channels<br/>C_out"]
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Two convs in series — so the weight from upstream neuron `i` to downstream
    neuron `j` is really a *path* through some bottleneck unit `b`. Step 2
    collapses that whole path into one `k × k` matrix.
    """)
    return


@app.cell
def _(fv, mo, model):
    block = mo.ui.dropdown(
        options=fv.inception_blocks(model),
        value="inception4b",
        label="Inception block",
    )
    block
    return (block,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2 — collapse the bottleneck into one effective kernel

    Two steps turn the branch's two convs into a single `k × k` weight per
    neuron pair:

    1. **Fold BatchNorm in.** At inference each conv is followed by a
       BatchNorm — a fixed per-channel scale-and-shift. Folding it back into
       the conv weight (`W' = W · γ / √(var + ε)`) gives the weight the
       network *actually applies*, with no separate BN step.

    2. **Multiply out the bottleneck.** Compose the `1×1` and `k×k` convs
       across the bottleneck axis `b`:

    $$W_\text{eff}[j, i, u, v] = \sum_b W_{1\times1}[b, i]\;\cdot\;W_{k\times k}[j, b, u, v]$$

    The result `W_eff` has shape `[C_out, C_in, k, k]` — one `k × k` matrix
    for **every** (upstream `i`, downstream `j`) pair in the block.
    """)
    return


@app.cell
def _(block, fv, model):
    effective = fv.effective_kernel(model, block.value, "branch2")
    return (effective,)


@app.cell
def _(block, effective, mo):
    _out, _in_, _k, _ = effective.shape
    mo.md(
        f"""
        For **`{block.value}.branch2`**, `W_eff` has shape
        `{tuple(effective.shape)}` = `[C_out={_out}, C_in={_in_}, k={_k}, k={_k}]`.

        That is **{_out} × {_in_} = {_out * _in_}** separate `{_k}×{_k}` weight
        matrices — one per neuron pair. Pick a pair below.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3 — one connection, up close

    Choose a downstream neuron `j` and an upstream neuron `i`. The matrix
    below is *the* weight between them — read it as a little spatial stamp
    over the receptive field.
    """)
    return


@app.cell
def _(effective, mo):
    _out, _in_, _k, _ = effective.shape
    downstream = mo.ui.slider(
        start=0, stop=_out - 1, value=100, label="Downstream neuron j", show_value=True
    )
    upstream = mo.ui.slider(
        start=0, stop=_in_ - 1, value=50, label="Upstream neuron i", show_value=True
    )
    mo.vstack([downstream, upstream])
    return downstream, upstream


@app.cell
def _(downstream, effective, fv, mo, show_weight_matrix, upstream):
    _w = fv.weight_matrix(effective, downstream.value, upstream.value)
    mo.vstack(
        [
            show_weight_matrix(
                _w, title=f"weight  i={upstream.value} → j={downstream.value}"
            ),
            mo.md(
                f"Each cell is a relative spatial offset. **Red (+)** — upstream "
                f"neuron `{upstream.value}` firing there **excites** downstream "
                f"neuron `{downstream.value}`; **blue (−)** — it **inhibits** it. "
                f"A near-zero matrix means these two neurons are barely connected."
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4 — every upstream neuron feeding one downstream neuron

    Fixing the downstream neuron `j`, here are the `k × k` weights from the
    first 25 upstream neurons at once — the panel the Distill weight explorer
    renders. Scanning it shows *which* upstream neurons drive `j`, and with
    what spatial pattern.
    """)
    return


@app.cell
def _(downstream, effective, mo, show_weight_grid):
    mo.vstack(
        [
            mo.md(f"### Upstream contributions into `j = {downstream.value}`"),
            show_weight_grid(effective, downstream.value, n_upstream=25),
        ]
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **From weights to a *circuit*.** A weight matrix on its own is just
            numbers. It becomes a **circuit** only once you know *what the two
            neurons detect*. Use notebook 01's scout (or notebook 03's feature
            visualisation) to identify upstream `i` and downstream `j`; then a
            strong positive `W[j, i]` reads as *"feature `i` helps build feature
            `j`"* — for example, edge detectors at the right offsets summing
            into a curve detector. That feature-to-feature wiring **is** Claim 2.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaway — Claim 2 (the wiring)

    - A weight in a CNN circuit is a **spatial pattern**, not a scalar.
    - An Inception branch is a **bottleneck** of two convs; folding in
      BatchNorm and multiplying them out yields one effective `k × k` kernel
      per neuron pair.
    - Sign = influence: **red excites, blue inhibits**, position = offset in
      the receptive field.
    - A weight only becomes a **circuit** once paired with the *features* it
      connects — that is the bridge from Claim 1 to Claim 2.

    Notebook **01** showed what a feature *is*; this notebook showed how
    features *connect*; notebook **03** synthesises the image a feature wants —
    the second, independent line of evidence for Claim 1.
    """)
    return


if __name__ == "__main__":
    app.run()

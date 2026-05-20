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
    # 4 — A *circuit*: features + weights together

    Based on Distill's *Circuits* thread —
    [*Zoom In*](https://distill.pub/2020/circuits/zoom-in/) and
    [*Visualizing Weights*](https://distill.pub/2020/circuits/visualizing-weights/).

    ## Where this sits — Claim 2, in full

    Notebook **01** showed what a feature *is* (its dataset crops). Notebook
    **02** showed the *weights* between neurons (`k × k` red/blue stamps) — but
    a weight on its own is just numbers. This notebook composes the two: a
    **circuit** is a downstream feature plus the upstream features that feed
    it, joined by the weights.

    The claim made visible here is concrete: **this downstream feature is
    built from these specific upstream features, in these specific spatial
    patterns.** Each row below reads like one term in a sum — *that* feature,
    weighted *this* way, pushes the downstream neuron up (or down) by *this
    much*.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **What this is — and isn't.** This is a **one-hop slice** through
            one branch of one Inception block: the *immediately previous*
            block's contribution into `branch2` of the chosen downstream
            block. The Distill articles also build longer arcs (oriented-edges
            → curves → 3-D shape) — those are several of these slices chained
            together. We also keep notebook 02's two simplifications: the
            ReLU between the two convs inside `branch2` is ignored, and only
            `branch2` is visualised (one of the block's four parallel paths).
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
                    "Circuit": mo.md(
                        "A small group of **features connected by weights** "
                        "that together compute something — a curve detector "
                        "built from edge detectors, an eye detector built from "
                        "fur and curve detectors. Claim 2 of the Circuits "
                        "papers is that circuits, like features, can be "
                        "understood."
                    ),
                    "One-hop slice": mo.md(
                        "The cut we draw: pick one downstream neuron and show "
                        "the **immediately preceding** layer's contributions "
                        "into it. Chain several hops and you have one of the "
                        "multi-layer arcs the articles draw."
                    ),
                    "Excitatory / inhibitory inputs": mo.md(
                        "Rank the upstream channels by the **signed sum** of "
                        "their `k × k` weight into the downstream neuron. Most "
                        "positive sums = excitatory inputs (build the feature "
                        "up); most negative = inhibitory inputs (features the "
                        "downstream neuron explicitly *rejects*)."
                    ),
                    "Branch2": mo.md(
                        "An Inception block has four parallel branches: "
                        "`1×1`, `1×1 → 3×3`, `1×1 → 5×5`, `pool → 1×1`. "
                        "`branch2` is the bottleneck-`3×3` path — the only "
                        "branch whose spatial weights are non-trivial, so the "
                        "only one whose *circuit shape* is interesting to draw."
                    ),
                    "Signed sum": mo.md(
                        "A scalar summary of a `k × k` weight: `sum(W)`. "
                        "Captures the **net push** of an upstream channel — a "
                        "rough ordering for picking the top inputs to show. "
                        "The matrix itself, not the sum, holds the *spatial* "
                        "story."
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _():
    import torch

    import feature_viz as fv
    from feature_viz.dataset_examples import (
        compute_dataset_activations,
        load_bundled_images,
        top_examples,
    )
    from feature_viz.plotting import show_circuit_rows, show_image_grid

    return (
        compute_dataset_activations,
        fv,
        load_bundled_images,
        show_circuit_rows,
        show_image_grid,
        top_examples,
        torch,
    )


@app.cell
def _(fv):
    # Forward passes over the bundled images — the runtime probe picks the device.
    device = fv.best_device()
    model = fv.load_inception(device)
    blocks = fv.inception_blocks(model)
    return blocks, model


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — pick the circuit

    Choose a downstream Inception block. The notebook reads its `branch2` and
    asks: which channels of the **previous** Inception block feed each of its
    outputs? Early blocks (`inception3b`, `inception4a`) are the easiest to
    read — upstream and downstream features are both visually simple (edges,
    curves, colour) so the wiring story is short. Deeper blocks reveal richer
    compositions (parts → objects), but the features themselves get harder to
    name.
    """)
    return


@app.cell
def _(blocks, mo):
    # Drop inception3a — its predecessor isn't another Inception block, so the
    # adjacent-block framing doesn't apply.
    options = [b for b in blocks if blocks.index(b) > 0]
    downstream_block = mo.ui.dropdown(
        options=options, value="inception3b", label="Downstream block"
    )
    downstream_block
    return (downstream_block,)


@app.cell
def _(blocks, downstream_block, mo):
    _idx = blocks.index(downstream_block.value)
    upstream_block_name = blocks[_idx - 1]
    mo.md(
        f"Circuit slice: **`{upstream_block_name}`** → "
        f"**`{downstream_block.value}.branch2`**."
    )
    return (upstream_block_name,)


@app.cell
def _(downstream_block, fv, model):
    effective = fv.effective_kernel(model, downstream_block.value, "branch2")
    return (effective,)


@app.cell
def _(effective, mo):
    _out, _in_, _k, _ = effective.shape
    mo.md(
        f"For this slice, `W_eff` has shape `{tuple(effective.shape)}` — "
        f"**{_out}** downstream branch2 neurons, **{_in_}** upstream channels, "
        f"each connected by a **{_k}×{_k}** matrix."
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2 — pick the downstream neuron `j`

    Type a `branch2`-internal channel index. Channel numbers are
    model-specific (the standard caveat from notebook 01), so the productive
    move is to try a few — when you land on one whose downstream feature is
    visually recognisable, its circuit will be easier to read. The default is
    a reasonable starting point, not a special index.
    """)
    return


@app.cell
def _(effective, mo):
    j = mo.ui.number(
        start=0,
        stop=effective.shape[0] - 1,
        value=50,
        label="Downstream neuron j  (in branch2)",
    )
    j
    return (j,)


@app.cell
def _(
    compute_dataset_activations,
    downstream_block,
    load_bundled_images,
    model,
    upstream_block_name,
):
    # One batched forward pass each — re-runs only when the block selection
    # changes (not on every `j` change).
    samples = load_bundled_images()
    da_upstream = compute_dataset_activations(model, upstream_block_name, samples)
    da_downstream = compute_dataset_activations(
        model, f"{downstream_block.value}.branch2", samples
    )
    return da_downstream, da_upstream


@app.cell
def _(effective, j, torch):
    # Rank upstream channels by net push into downstream neuron j: most positive
    # = strongest excitatory, most negative = strongest inhibitory. The matrix
    # shape (not the sum) carries the spatial story; the sum is just the order.
    signed_sum = effective[int(j.value)].sum(dim=(1, 2))  # [C_in]
    order = torch.argsort(signed_sum, descending=True)
    K = 5
    top_excite = order[:K].tolist()
    top_inhibit = order[-K:].flip(dims=[0]).tolist()  # most negative first
    return K, top_excite, top_inhibit


@app.cell
def _(K, downstream_block, j, mo, upstream_block_name):
    mo.md(
        f"## Step 3 — the circuit\n\n"
        f"Ranking the upstream channels of **`{upstream_block_name}`** by their "
        f"signed contribution into **`{downstream_block.value}.branch2 : ch "
        f"{int(j.value)}`** — the **{K} most excitatory** and the **{K} most "
        f"inhibitory** are shown below, each next to the upstream neuron's "
        f"feature crops.\n\n"
        f"To also see each upstream neuron's **synthesised feature** (notebook "
        f"03's gradient-ascent thumbnail, a second representation of what the "
        f"neuron detects), click the button below. The render is coarse "
        f"(128 steps, 96 px — visibly noisier than notebook 03's polished "
        f"version) and cached: ≈20 s the first time, then ≈2 s per new channel "
        f"as you change `j`."
    )
    return


@app.cell
def _():
    # Persistent in-memory cache, shared across the two row cells. No inputs ->
    # this cell runs once per session; subsequent updates happen by mutation
    # below. Snapshots are returned from the render cell to drive reactivity.
    upstream_synth_cache: dict[tuple[str, int], object] = {}
    return (upstream_synth_cache,)


@app.cell
def _(mo):
    synth_button = mo.ui.run_button(
        label="Render upstream synth thumbnails (~20 s for 10 channels)"
    )
    synth_button
    return (synth_button,)


@app.cell
def _(
    fv,
    model,
    synth_button,
    top_excite,
    top_inhibit,
    upstream_block_name,
    upstream_synth_cache: dict[tuple[str, int], object],
):
    # When the button is clicked (synth_button.value transitions True once),
    # render any channels not already cached. The cache survives across j
    # changes, so re-clicking after changing j only renders the *new* channels.
    if synth_button.value:
        for _i in top_excite + top_inhibit:
            _key = (upstream_block_name, _i)
            if _key not in upstream_synth_cache:
                _r = fv.render_neuron(
                    model,
                    upstream_block_name,
                    _i,
                    steps=128,
                    size=96,
                    seed=0,
                )
                upstream_synth_cache[_key] = _r.image
    # Snapshot at every cell run so downstream row cells re-render via marimo's
    # reactive graph (mutations to upstream_synth_cache alone don't trigger).
    synth_snapshot = dict(upstream_synth_cache)
    return (synth_snapshot,)


@app.cell
def _(da_downstream, downstream_block, j, mo, show_image_grid, top_examples):
    _exs = top_examples(da_downstream, int(j.value), top_k=6, crop_size=96)
    _crops = [(f"{e.name}\n{e.score:.1f}", e.crop) for e in _exs]
    mo.vstack(
        [
            mo.md(
                f"### Downstream feature — `{downstream_block.value}.branch2 : "
                f"ch {int(j.value)}`\n\n"
                f"What this downstream neuron *detects*, read off its peak "
                f"dataset crops. The rows below decompose what builds it."
            ),
            show_image_grid(_crops, ncols=6),
        ]
    )
    return


@app.cell
def _(
    da_upstream,
    effective,
    fv,
    j,
    mo,
    show_circuit_rows,
    synth_snapshot,
    top_examples,
    top_excite,
    upstream_block_name,
):
    _rows = []
    _thumbs = []
    for _i in top_excite:
        _w = fv.weight_matrix(effective, int(j.value), _i)
        _exs = top_examples(da_upstream, _i, top_k=4, crop_size=64)
        _rows.append((_i, float(_w.sum()), _w, [e.crop for e in _exs]))
        _thumbs.append(synth_snapshot.get((upstream_block_name, _i)))
    _has_any_thumb = any(t is not None for t in _thumbs)
    mo.vstack(
        [
            mo.md("### Excitatory inputs — features that **build** this neuron"),
            show_circuit_rows(
                _rows,
                n_crops=4,
                synth_thumbnails=_thumbs if _has_any_thumb else None,
            ),
        ]
    )
    return


@app.cell
def _(
    da_upstream,
    effective,
    fv,
    j,
    mo,
    show_circuit_rows,
    synth_snapshot,
    top_examples,
    top_inhibit,
    upstream_block_name,
):
    _rows = []
    _thumbs = []
    for _i in top_inhibit:
        _w = fv.weight_matrix(effective, int(j.value), _i)
        _exs = top_examples(da_upstream, _i, top_k=4, crop_size=64)
        _rows.append((_i, float(_w.sum()), _w, [e.crop for e in _exs]))
        _thumbs.append(synth_snapshot.get((upstream_block_name, _i)))
    _has_any_thumb = any(t is not None for t in _thumbs)
    mo.vstack(
        [
            mo.md("### Inhibitory inputs — features the neuron **rejects**"),
            show_circuit_rows(
                _rows,
                n_crops=4,
                synth_thumbnails=_thumbs if _has_any_thumb else None,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What you're reading

    Each row in the two panels above is one **upstream → downstream** wiring:

    - **Left** — the `k × k` weight matrix from upstream channel `i` into the
      downstream neuron `j`. Read as a spatial stamp: red entries *excite*,
      blue *inhibit*. The number after the channel label is the **signed sum**
      of the matrix — its net push, which is how the rows are ranked.
    - **Right** — that same upstream neuron's **feature**: its peak dataset
      crops, exactly the panel notebook 01 draws.

    Put together, each row says: *"this feature, weighted this way,
    contributes this much to the downstream feature above."*

    A clean monosemantic circuit shows excitatory inputs whose features look
    *related to* the downstream feature (e.g. oriented edges of compatible
    orientation feeding into a curve detector), and inhibitory inputs that
    look *opposed* (edges in the wrong orientation, or competing features
    being actively suppressed). A polysemantic downstream neuron's inputs
    tend to be a grab-bag — informative in its own way: it shows several
    sub-features sharing one axis.
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Tip.** If the circuit looks like noise, scout a different `j` —
            many channels are polysemantic or weakly tuned, and the cleanest
            stories live at a small fraction of indices. Notebook 01's scout
            on the downstream block (at the appropriate branch2 offset) is
            the principled way to find them; trying a handful of `j` values
            by hand also works.
            """
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaway — Claim 2 in full

    - A **circuit** is the composition of Claim 1 (features) with the wiring
      from notebook 02 (weights). Reading the two together — *what feeds
      what, via what spatial pattern* — is what makes the wiring meaningful.
    - One downstream neuron is a **weighted combination** of upstream
      features: excitatory inputs build it up, inhibitory inputs suppress
      directions that don't belong.
    - This is a **one-hop** slice. Chain several slices and you get the
      multi-layer arcs the Circuits articles draw — e.g. oriented edges →
      curves → 3-D-shape grouping.
    - Channel indices are still model-specific (notebook 01's caveat); the
      productive move is to **scout** `j` until the downstream feature is
      recognisable, then read its circuit.

    Notebooks **01** and **03** show what a feature *is*; **02** shows the raw
    wiring; this notebook is where the two meet — and that meeting is what
    the Circuits papers call *understanding*.
    """)
    return


if __name__ == "__main__":
    app.run()

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
    # 1 — What a "neuron" is, and what it detects

    Based on Distill's *Circuits* thread — [*Zoom In*](https://distill.pub/2020/circuits/zoom-in/)
    and [*Early Vision*](https://distill.pub/2020/circuits/early-vision/).

    ## The one idea

    A "neuron" in a CNN is **not an object**. It is an *address* — a pair
    `(layer, channel)`. Nothing is stored at that address; the neuron only
    takes a *value* when an image flows through the network.

    ## Two things people confuse

    | Question | What you look at | Changes with the input image? |
    | --- | --- | --- |
    | **Where** does the neuron fire? | its **activation map** `act[:, channel]` | **yes** |
    | **What** does the neuron detect? | **dataset crops** — patches it fires on | **no** — that *is* the neuron |

    The activation map is a *response* to one image. The dataset crops are the
    neuron's *feature* — its identity. The curve / edge / texture pictures in
    the Distill articles are the **feature**, so they come from the crops in
    Step 4 — never from the activation heatmap.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## What *Zoom In* claims

    The paper builds **three claims**, each resting on the one before:

    1. **Features** — features (a curve detector, an edge detector, …) are the
       fundamental unit of a network, and they *can be understood*.
    2. **Circuits** — features are wired together by weights into **circuits**
       that compute something meaningful.
    3. **Universality** — the same features and circuits recur across networks.

    These three notebooks cover **Claim 1 in full** and open **Claim 2**:

    | Notebook | Covers |
    | --- | --- |
    | **01** (here) | Claim 1 — what a feature *is*, seen via dataset examples |
    | **02** | Claim 2 — the weights that wire features into circuits |
    | **03** | Claim 1 — *feature visualisation*, a second, independent way to see a feature |

    Claim 3 (universality) is left to the articles. This notebook is Claim 1:
    *what a feature is, and how to see one.*
    """)
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                "**Background** — the paper assumes these. Expand any that are "
                "new to you; skip if you already know them:"
            ),
            mo.accordion(
                {
                    "Neural network": mo.md(
                        "A stack of simple numerical operations ('layers') with "
                        "millions of adjustable numbers ('weights'). Given an "
                        "image it outputs, here, a guess at which of 1000 classes "
                        "it shows."
                    ),
                    "Convolutional network (CNN) & layer": mo.md(
                        "A **layer** is one processing stage. A **CNN** is a "
                        "network whose layers are *convolutions* — the standard "
                        "design for images. InceptionV1 stacks ~10 Inception "
                        "blocks; each block is a layer in our sense."
                    ),
                    "Convolution / filter": mo.md(
                        "A small grid of weights (a **filter**) slid across the "
                        "image; at each position it produces one number. One "
                        "filter, applied everywhere, yields one **channel** — and "
                        "one channel is one neuron."
                    ),
                    "Tensor": mo.md(
                        "A multi-dimensional array of numbers. An image is a "
                        "`3 x 224 x 224` tensor; a layer's output is the 4-D "
                        "tensor `[batch, channels, height, width]`."
                    ),
                    "Weights vs. activations": mo.md(
                        "**Weights** are the network's fixed, learned numbers — "
                        "the same for every image. **Activations** are what a "
                        "layer *outputs* for one specific image — they change with "
                        "the input. A neuron's activation map is activations."
                    ),
                    "Training & pretrained model": mo.md(
                        "*Training* tunes the weights on labelled data until the "
                        "network classifies well — we do **not** train here. We "
                        "load an already-trained ('pretrained') model and only "
                        "inspect it."
                    ),
                    "ImageNet": mo.md(
                        "The dataset of ~1.2 million labelled photos across 1000 "
                        "classes that InceptionV1 was trained on. Its statistics "
                        "shape every feature the network learned."
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.mermaid(
        """
        graph LR
            IMG["input image<br/>3 x 224 x 224"] --> NET["InceptionV1<br/>frozen, ImageNet-trained"]
            NET --> ACT["activation tensor<br/>at a chosen layer<br/>B x C x H x W"]
            ACT --> SLICE["channel c:<br/>act[:, c] = an H x W map"]
            SLICE --> NEU(["neuron = (layer, c)"])
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## A neuron, said more carefully: a *direction*

    At one spot in a layer the network outputs `C` numbers — picture them as an
    **arrow** in a space with `C` axes. Each **neuron is one axis** of that
    space. The paper's precise Claim 1 is that a **feature is a direction** —
    most often a single neuron's axis, occasionally a diagonal mix of a few.

    Why mostly neurons? The network's nonlinearity (ReLU) acts on each axis
    separately, which makes the neuron axes a *privileged* set of directions. In
    early vision most features really are single neurons — so these notebooks
    treat **neuron = feature**. The honest exception is a **polysemantic**
    neuron (Step 4).
    """)
    return


@app.cell
def _(mo, show_feature_directions):
    mo.vstack(
        [
            show_feature_directions(),
            mo.md(
                "*Only two axes are drawn; a real layer has hundreds. The blue "
                "axes are neurons; the crimson arrow is a feature that is a "
                "direction — not aligned to any single neuron.*"
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("**Concepts** — expand any term you want defined:"),
            mo.accordion(
                {
                    "Receptive field": mo.md(
                        "The patch of the **input image** that can influence one "
                        "cell of a neuron's activation map. Early layers have a "
                        "*small* receptive field (edges, textures); deep layers a "
                        "*large* one (whole objects). The dataset crops in Step 4 "
                        "are cut to approximate it."
                    ),
                    "Channel & neuron": mo.md(
                        "A convolutional layer outputs `C` stacked feature maps. "
                        "Each map is one **channel**, and one channel *is* one "
                        "**neuron** — the same small filter applied at every "
                        "spatial position."
                    ),
                    "Activation map": mo.md(
                        "The `H x W` grid of numbers one neuron produces for one "
                        "input — its **response**. Bright = strong firing. It "
                        "changes when the input changes."
                    ),
                    "Feature vs. response": mo.md(
                        "**Feature** = *what* the neuron detects (intrinsic — its "
                        "identity). **Response** = *where* it fired on one given "
                        "image. The Distill curve/edge pictures are features."
                    ),
                    "Feature — a direction": mo.md(
                        "The paper's precise Claim 1: a feature is a **direction** "
                        "in a layer's activation space. A neuron is one *axis* of "
                        "that space — the most common kind of feature, but not the "
                        "only kind."
                    ),
                    "Privileged basis": mo.md(
                        "The ReLU nonlinearity acts on each coordinate (neuron) "
                        "separately. That singles the neuron axes out as "
                        "*privileged* directions — which is why looking at "
                        "individual neurons is a sound first move."
                    ),
                    "Monosemantic vs. polysemantic": mo.md(
                        "A **monosemantic** neuron detects one coherent thing — a "
                        "clean feature. A **polysemantic** neuron fires for several "
                        "*unrelated* things at once (its dataset crops are a "
                        "grab-bag). Polysemantic neurons are the honest caveat to "
                        "'neuron = feature'."
                    ),
                    "Selectivity": mo.md(
                        "How picky a neuron is. A *selective* neuron fires on few "
                        "inputs (a sharp detector); a *broad* one fires on almost "
                        "everything, so its map just traces image structure."
                    ),
                    "InceptionV1 / GoogLeNet": mo.md(
                        "A 2014 convolutional image classifier. We use "
                        "`torchvision`'s ImageNet-trained re-implementation — "
                        "faithful in structure, but with its own channel numbering."
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Channel numbers here do not match the Distill articles.** This is
            `torchvision`'s GoogLeNet — a re-implementation with its *own* channel
            ordering. `inception3b:100` here is **not** the article's famous
            `mixed3b:100`. So you cannot just type the article's number — you have
            to **scout** for an interesting neuron (Step 3). Curve detectors very
            likely exist in this model too, just at unknown indices.
            """
        ),
        kind="warn",
    )
    return


@app.cell
def _():
    import io

    import feature_viz as fv
    from feature_viz.dataset_examples import (
        channel_cards,
        compute_dataset_activations,
        load_bundled_images,
        top_examples,
    )
    from feature_viz.neuron import gray_placeholder
    from feature_viz.plotting import (
        show_activation_overlay,
        show_activation_ranking,
        show_feature_directions,
        show_image_grid,
    )
    from PIL import Image

    return (
        Image,
        channel_cards,
        compute_dataset_activations,
        fv,
        gray_placeholder,
        io,
        load_bundled_images,
        show_activation_overlay,
        show_activation_ranking,
        show_feature_directions,
        show_image_grid,
        top_examples,
    )


@app.cell
def _(fv):
    # Forward passes only in this notebook — the runtime probe picks the device.
    device = fv.best_device()
    model = fv.load_inception(device)
    return (model,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — pick a layer and an image

    An **Inception layer** is one stage of InceptionV1. Early layers
    (`inception3a/b`) see small patches and detect simple things — edges,
    curves, colour. Deeper layers (`inception4*`, `inception5*`) see wider
    regions and detect parts and objects.

    The image is optional: with none, a flat gray placeholder is used (its
    activation map will be near-empty — that is itself instructive).
    """)
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.mermaid(
                """
                graph LR
                    IN["input<br/>image"] --> E["early layers<br/>inception3a/3b<br/>edges · curves · colour"]
                    E --> M["mid layers<br/>inception4a-4e<br/>textures · parts"]
                    M --> D["deep layers<br/>inception5a/5b<br/>objects"]
                    D --> OUT["1000-class<br/>prediction"]
                """
            ),
            mo.md(
                "*Features grow more complex with depth — that progression is the "
                "whole subject of* Early Vision*. The layer you pick decides which "
                "kind of feature you will meet.*"
            ),
        ]
    )
    return


@app.cell
def _(fv, mo, model):
    layer = mo.ui.dropdown(
        options=fv.inception_blocks(model),
        value="inception3b",
        label="Inception layer",
    )
    image_upload = mo.ui.file(
        kind="button", label="Image (optional — gray placeholder otherwise)"
    )
    mo.hstack([layer, image_upload], justify="start", gap=2)
    return image_upload, layer


@app.cell
def _(Image, gray_placeholder, image_upload, io):
    if image_upload.value:
        pil_image = Image.open(io.BytesIO(image_upload.value[0].contents)).convert(
            "RGB"
        )
    else:
        pil_image = gray_placeholder()
    return (pil_image,)


@app.cell
def _(mo, pil_image):
    mo.image(
        pil_image,
        width=256,
        caption="Your input image — resized to 256, centre-cropped to 224, and "
        "normalised before the forward pass.",
    )
    return


@app.cell
def _(fv, layer, model, pil_image):
    x = fv.preprocess_image(pil_image)
    activation = fv.neuron_activation(model, layer.value, x)
    return (activation,)


@app.cell
def _(activation, layer, mo):
    _b, _c, _h, _w = activation.shape
    mo.md(
        rf"""
        ## Step 2 — the activation tensor

        Your image produced the tensor `{tuple(activation.shape)}` at
        **`{layer.value}`** — its axes are `[batch, channels, height, width]`:

        - **batch = {_b}** — one image in, one set of activations out.
        - **channels = {_c}** — this layer holds **{_c} neurons**. Channel `c`
          *is* neuron `c`.
        - **height × width = {_h} × {_w}** — each neuron is not one number but a
          small grid: it is applied at every one of {_h}×{_w} = {_h * _w}
          positions, so it has an opinion at each.

        "Neuron `{layer.value}:c`" is the slice `act[:, c, :, :]` — one
        **{_h}×{_w}** scalar map. That map is what Step 4 calls the *response*.
        """
    )
    return


@app.cell
def _(activation):
    # Print the raw activation tensor so its shape/dtype/values are visible
    # alongside the prose above. PyTorch already truncates large tensors.
    print(activation)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3 — the channel scout: find an interesting neuron

    Because the channel numbers are arbitrary (see the warning above), the
    honest way to find a detector is to **look**. Below is one *card* per
    channel: the small image patch where that channel fires hardest across a
    bundled set of ~96 photos — its **receptive-field crop**.

    Scan the grid. Edge / curve / texture detectors jump out as repetitive
    crops; object-ish channels show recognisable things. Note an index you
    like and type it into Step 4. Use the page slider to walk all channels.
    """)
    return


@app.cell
def _(compute_dataset_activations, layer, load_bundled_images, model):
    samples = load_bundled_images()
    # One batched forward pass over the bundled images; redone only on layer change.
    da = compute_dataset_activations(model, layer.value, samples)
    return (da,)


@app.cell
def _(channel_cards, da):
    cards = channel_cards(da, crop_size=64)
    return (cards,)


@app.cell
def _():
    # One source of truth for scout pagination: the slider's page count and the
    # per-page slice must use the same value, or the slider runs past the last
    # populated page and shows empty grids.
    SCOUT_PER_PAGE = 40
    return (SCOUT_PER_PAGE,)


@app.cell
def _(SCOUT_PER_PAGE, cards, mo):
    _n_pages = (len(cards) + SCOUT_PER_PAGE - 1) // SCOUT_PER_PAGE
    page = mo.ui.slider(
        start=0, stop=max(1, _n_pages - 1), value=0, label="Scout page", show_value=True
    )
    page
    return (page,)


@app.cell
def _(SCOUT_PER_PAGE, cards, page, show_image_grid):
    _start = page.value * SCOUT_PER_PAGE
    _slice = cards[_start : _start + SCOUT_PER_PAGE]
    show_image_grid([(f"ch {c.channel}", c.crop) for c in _slice], ncols=8)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4 — inspect one neuron

    Type the channel index you want to study. The panel below shows it two
    ways — its **response** to *your* image (left) and its **feature**
    (right): the same neuron's peak crops across the bundled dataset.
    """)
    return


@app.cell
def _(da, mo):
    channel = mo.ui.number(
        start=0,
        stop=da.scores.shape[1] - 1,
        value=100,
        label="Neuron (channel index)",
    )
    channel
    return (channel,)


@app.cell
def _(
    activation,
    channel,
    da,
    fv,
    layer,
    mo,
    pil_image,
    show_activation_overlay,
    show_image_grid,
    top_examples,
):
    _c = int(channel.value)
    _map = fv.channel_slice(activation, _c)
    _stats = fv.neuron_stats(_map)
    _display = fv.preprocess_display(pil_image)
    _examples = top_examples(da, _c, top_k=8, crop_size=96)
    _crops = [(f"{e.name}\n{e.score:.1f}", e.crop) for e in _examples]
    mo.vstack(
        [
            mo.md(f"### Neuron `{layer.value}:{_c}` — response vs. feature"),
            mo.hstack(
                [
                    show_activation_overlay(
                        _display, _map, title="RESPONSE — where it fires on YOUR image"
                    ),
                    show_image_grid(_crops, ncols=4),
                ],
                justify="start",
                gap=2,
            ),
            mo.md(
                f"**Left — the response.** Your input image with neuron `{_c}`'s "
                f"activation glowing on top (bright = strong firing); the **cyan "
                f"box** marks its peak — the receptive-field patch it reacts to "
                f"most. Peak `{_stats.max:+.2f}` at `(y={_stats.argmax_yx[0]}, "
                f"x={_stats.argmax_yx[1]})`, mean `{_stats.mean:+.2f}`. Change the "
                f"image and the glow moves.\n\n"
                f"**Right — the feature.** The same kind of patch — the "
                f"receptive-field crop at the firing peak — but for the bundled "
                f"dataset images that drive this neuron hardest. That is *what it "
                f"detects*, independent of your image. The cyan box on the left "
                f"and these crops are the same thing, on different images: a "
                f"clean detector shows one repeated motif."
            ),
        ]
    )
    return


@app.cell
def _(activation, channel, da, fv, mo, show_activation_ranking):
    _c = int(channel.value)
    _user_score = float(fv.channel_slice(activation, _c).max())
    mo.vstack(
        [
            mo.md(
                "### The bridge — where *your* image ranks\n\n"
                "Both panels above describe the same neuron, but the response and "
                "the feature are different things. Here is the link: every bundled "
                "image ranked by how hard it fires this neuron, with **your image** "
                "(★, crimson) slotted in. Change your image and its bar moves."
            ),
            show_activation_ranking(da.names, da.scores[:, _c], _user_score),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5 — monosemantic, or polysemantic?

    Look back at the crop grid (the **feature**, right panel of Step 4) and ask
    one question: *do the crops all show the same kind of thing?*

    - **Monosemantic** — every crop shows one coherent motif (all curves, all
      fur, all eyes). The neuron is a clean, single feature. This is the case
      the paper's Claim 1 is built on.
    - **Polysemantic** — the crops are an unrelated grab-bag (a dog ear, a car
      wheel, some text). Several distinct features are *sharing* one neuron.

    Polysemantic neurons are the honest caveat to "neuron = feature": the
    fundamental unit is the **feature** (a direction), and it does not always
    line up with a single neuron. Scout a few channels — you will find both
    kinds. Selective-looking channels in the scout tend to be the cleaner,
    monosemantic features.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaway — Claim 1

    - A neuron is the address `(layer, channel)`; more precisely a feature is a
      **direction** in activation space, and a neuron is one axis of it.
    - Its **response** is a per-image activation map (*where* it fires); its
      **feature** is intrinsic (*what* it detects), read off the dataset crops.
    - Some neurons are **polysemantic** — the feature, not the neuron, is the
      true unit. That is the heart of Claim 1.
    - Channel numbers are model-specific; **scout**, do not guess.

    Next: notebook **03** gives a *second, independent* way to see a feature —
    synthesising the image that drives it hardest; agreement between the two is
    what justifies a label like "curve detector". Notebook **02** opens Claim 2:
    the *weights* that wire features into circuits.
    """)
    return


if __name__ == "__main__":
    app.run()

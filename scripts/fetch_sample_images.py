"""Download the curated sample-image set bundled with feature-viz.

These ~96 images (one per varied ImageNet class) back the dataset-examples
depiction in notebook 01 — the panel that shows what a neuron "looks like" as a
blend of the images that activate it most.

Source: the `imagenet-sample-images` repository (one representative JPEG per
ImageNet-1k class). The images are ImageNet samples, bundled here for
non-commercial educational use only.

Run once to (re)populate `src/feature_viz/sample_images/`:

    uv run python scripts/fetch_sample_images.py
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

RAW = "https://raw.githubusercontent.com/EliSchwartz/imagenet-sample-images/master"
MAX_SIDE = 384  # longest edge after downscaling — keeps the bundle small

# 96 deliberately varied classes so different neurons light up — a broad set
# gives the channel scout and dataset crops plenty to work with.
SOURCE_FILES = [
    "n01443537_goldfish.JPEG",
    "n01614925_bald_eagle.JPEG",
    "n01833805_hummingbird.JPEG",
    "n02056570_king_penguin.JPEG",
    "n02099601_golden_retriever.JPEG",
    "n02123045_tabby.JPEG",
    "n02119022_red_fox.JPEG",
    "n02391049_zebra.JPEG",
    "n02504458_African_elephant.JPEG",
    "n02279972_monarch.JPEG",
    "n02165456_ladybug.JPEG",
    "n02676566_acoustic_guitar.JPEG",
    "n03452741_grand_piano.JPEG",
    "n02690373_airliner.JPEG",
    "n04285008_sports_car.JPEG",
    "n04147183_schooner.JPEG",
    "n02980441_castle.JPEG",
    "n07745940_strawberry.JPEG",
    "n07873807_pizza.JPEG",
    "n09428293_seashore.JPEG",
    "n01644373_tree_frog.JPEG",
    "n01910747_jellyfish.JPEG",
    "n01944390_snail.JPEG",
    "n01806143_peacock.JPEG",
    "n02007558_flamingo.JPEG",
    "n01882714_koala.JPEG",
    "n02510455_giant_panda.JPEG",
    "n02132136_brown_bear.JPEG",
    "n02128385_leopard.JPEG",
    "n02480855_gorilla.JPEG",
    "n02268443_dragonfly.JPEG",
    "n01518878_ostrich.JPEG",
    "n02672831_accordion.JPEG",
    "n03495258_harp.JPEG",
    "n04398044_teapot.JPEG",
    "n04507155_umbrella.JPEG",
    "n03345487_fire_engine.JPEG",
    "n03792782_mountain_bike.JPEG",
    "n04310018_steam_locomotive.JPEG",
    "n02951358_canoe.JPEG",
    "n03888257_parachute.JPEG",
    "n03388043_fountain.JPEG",
    "n07749582_lemon.JPEG",
    "n07714990_broccoli.JPEG",
    "n07734744_mushroom.JPEG",
    "n07753275_pineapple.JPEG",
    "n07697313_cheeseburger.JPEG",
    "n07614500_ice_cream.JPEG",
    "n01484850_great_white_shark.JPEG",
    "n01914609_sea_anemone.JPEG",
    "n02317335_starfish.JPEG",
    "n01986214_hermit_crab.JPEG",
    "n01983481_American_lobster.JPEG",
    "n01818515_macaw.JPEG",
    "n01820546_lorikeet.JPEG",
    "n01843383_toucan.JPEG",
    "n01860187_black_swan.JPEG",
    "n01641577_bullfrog.JPEG",
    "n01669191_box_turtle.JPEG",
    "n01677366_common_iguana.JPEG",
    "n01770393_scorpion.JPEG",
    "n01774750_tarantula.JPEG",
    "n01784675_centipede.JPEG",
    "n02123394_Persian_cat.JPEG",
    "n02123597_Siamese_cat.JPEG",
    "n02125311_cougar.JPEG",
    "n02130308_cheetah.JPEG",
    "n02134084_ice_bear.JPEG",
    "n02326432_hare.JPEG",
    "n02342885_hamster.JPEG",
    "n02346627_porcupine.JPEG",
    "n02363005_beaver.JPEG",
    "n02398521_hippopotamus.JPEG",
    "n02423022_gazelle.JPEG",
    "n02437616_llama.JPEG",
    "n02444819_otter.JPEG",
    "n02480495_orangutan.JPEG",
    "n02701002_ambulance.JPEG",
    "n04146614_school_bus.JPEG",
    "n03384352_forklift.JPEG",
    "n04347754_submarine.JPEG",
    "n04266014_space_shuttle.JPEG",
    "n02787622_banjo.JPEG",
    "n03110669_cornet.JPEG",
    "n03372029_flute.JPEG",
    "n02769748_backpack.JPEG",
    "n02841315_binoculars.JPEG",
    "n02948072_candle.JPEG",
    "n03400231_frying_pan.JPEG",
    "n03481172_hammer.JPEG",
    "n04179913_sewing_machine.JPEG",
    "n03028079_church.JPEG",
    "n03160309_dam.JPEG",
    "n03457902_greenhouse.JPEG",
    "n03788195_mosque.JPEG",
    "n03956157_planetarium.JPEG",
]


def main() -> None:
    out_dir = (
        Path(__file__).resolve().parent.parent / "src" / "feature_viz" / "sample_images"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCE_FILES:
        stem = source.split("_", 1)[1].rsplit(".", 1)[
            0
        ]  # n01443537_goldfish.JPEG -> goldfish
        data = urlopen(f"{RAW}/{source}", timeout=30).read()  # noqa: S310 (trusted host)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        dst = out_dir / f"{stem}.jpg"
        img.save(dst, "JPEG", quality=85)
        print(f"  {dst.name:28s} {img.size}")
    print(f"saved {len(SOURCE_FILES)} images to {out_dir}")


if __name__ == "__main__":
    main()

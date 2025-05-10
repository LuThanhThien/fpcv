from __future__ import annotations

from pathlib import Path
from datetime import datetime
import random

from PIL import Image


def generate_multi_object_images(
    data_dir: Path | str,
    *,
    num_images: int = 100,
    max_objects: int = 5,
    img_mode: str = "L",          # keep original greyscale/binary look
) -> None:
    """
    Concatenate 2–`max_objects` single‑object images along a *random* axis
    (row‑wise or column‑wise) and save the composites in <data_dir>/augment.

    *Canvas size is determined **per composite** from the selected images*,
    not from the whole dataset.

    Parameters
    ----------
    data_dir : str | Path
        Folder that contains the original single‑object images.
    num_images : int, default 100
        How many concatenated images to create.
    max_objects : int, default 5
        Maximum number of source images to join in a single composite.
    img_mode : str, default "L"
        Pillow image mode used for loading + saving ("L"→8‑bit, "1"→1‑bit).
    """

    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise NotADirectoryError(data_dir)

    # gather all image files (skip the augment folder)
    img_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
    src_paths = [
        p for p in data_dir.iterdir()
        if p.suffix.lower() in img_exts and p.parent.name != "augment"
    ]
    if not src_paths:
        raise RuntimeError(f"No image files found in {data_dir}")

    out_dir = data_dir / "augment"
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx in range(num_images):
        k = random.randint(2, max_objects)
        chosen = random.sample(src_paths, k)
        random.shuffle(chosen)                       # random order

        # load all k images
        imgs = [Image.open(p).convert(img_mode) for p in chosen]
        widths, heights = zip(*(im.size for im in imgs))

        # randomly decide concat direction
        horizontal = random.choice([True, False])

        if horizontal:
            # ── canvas size: sum width, max height
            canvas_w = sum(widths)
            canvas_h = max(heights)
        else:
            # ── canvas size: max width, sum height
            canvas_w = max(widths)
            canvas_h = sum(heights)

        # build blank canvas and paste images one after another
        canvas = Image.new(img_mode, (canvas_w, canvas_h), 0)
        offset_x = offset_y = 0
        for im in imgs:
            canvas.paste(im, (offset_x, offset_y))
            if horizontal:
                offset_x += im.width            # advance in x
            else:
                offset_y += im.height           # advance in y

        # save
        fname = f"concat_{timestamp}_{idx:04d}_{k}imgs.png"
        canvas.save(out_dir / fname)
        
        if idx % 5 == 0:
            print(f"Save to {out_dir/fname} with {k} images")

    print(f"✓ Generated {num_images} concatenated images in {out_dir}")

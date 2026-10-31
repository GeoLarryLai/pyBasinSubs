# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Larry Syu-Heng Lai
"""Re-execute the notebooks in place and shrink their images.

    python tools/run_notebooks.py               # all notebooks
    python tools/run_notebooks.py 6 8 9         # by number

Policy: the notebooks in notebooks/ are the source of truth and are kept
*executed* (outputs embedded) so they read well on GitHub. To keep the repo
small, embedded PNG outputs and the figures in image/ are palette-quantized
(Pillow, 256 colours) after execution; figures are saved at 120 dpi.
"""
import base64
import glob
import io
import os
import sys

import nbformat
from nbclient import NotebookClient
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def shrink_png(raw: bytes) -> bytes:
    im = Image.open(io.BytesIO(raw))
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        q = im.quantize(256, method=Image.Quantize.FASTOCTREE)
    else:
        q = im.convert("RGB").quantize(256, method=Image.Quantize.MEDIANCUT)
    out = io.BytesIO(); q.save(out, "PNG", optimize=True)
    return out.getvalue() if len(out.getvalue()) < len(raw) else raw


def shrink_notebook(nb):
    for c in nb.cells:
        for o in c.get("outputs", []):
            d = o.get("data", {})
            if "image/png" in d:
                d["image/png"] = base64.b64encode(shrink_png(base64.b64decode(d["image/png"]))).decode()
    return nb


def shrink_images(names=None):
    for p in glob.glob(os.path.join(ROOT, "image", "*.png")):
        if names and not any(os.path.basename(p).startswith(f"nb{n}_") for n in names):
            continue
        raw = open(p, "rb").read(); new = shrink_png(raw)
        if len(new) < len(raw):
            open(p, "wb").write(new)


def main(args):
    nbs = sorted(glob.glob(os.path.join(ROOT, "notebooks", "pyBasinSubs_*.ipynb")))
    if args:
        nbs = [p for p in nbs if os.path.basename(p).split("_")[1] in args]
    for p in nbs:
        print("executing", os.path.basename(p), flush=True)
        nb = nbformat.read(p, as_version=4)
        NotebookClient(nb, timeout=3600, kernel_name="python3",
                       resources={"metadata": {"path": os.path.join(ROOT, "notebooks")}}).execute()
        nbformat.write(shrink_notebook(nb), p)
    shrink_images(args or None)


if __name__ == "__main__":
    main(sys.argv[1:])

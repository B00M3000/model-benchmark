"""Run-length encoding for instance masks.

Masks are kept only so the comparison video can be rendered after the fact
without re-running inference. Storing raw bitmaps for every frame of every
detection is far too large, and RLE on a binary mask is both compact and
cheap to decode.

Row-major, counts alternating starting with a run of zeros -- the same
convention COCO uses, minus the column-major transpose.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def encode(mask: np.ndarray) -> dict[str, Any]:
    """RLE-encode a 2-D boolean mask."""
    flat = np.asarray(mask, dtype=bool).ravel(order="C")
    if flat.size == 0:
        return {"size": list(mask.shape), "counts": []}

    # Indices where the value flips, turned into run lengths.
    changes = np.flatnonzero(np.diff(flat)) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    runs = np.diff(bounds)

    counts = runs.tolist()
    # The convention is that the first run is of zeros; if the mask starts
    # with a 1, lead with an empty zero-run.
    if flat[0]:
        counts = [0] + counts
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def decode(rle: dict[str, Any]) -> np.ndarray:
    """Inverse of :func:`encode`."""
    height, width = rle["size"]
    flat = np.zeros(height * width, dtype=bool)
    position = 0
    value = False
    for count in rle["counts"]:
        if value and count:
            flat[position : position + count] = True
        position += count
        value = not value
    return flat.reshape((height, width))


def area(rle: dict[str, Any]) -> int:
    """Number of set pixels, without materialising the mask."""
    return int(sum(rle["counts"][1::2]))

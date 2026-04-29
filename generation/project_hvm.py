"""Project HVM 3D object parameters to 2D pixel coordinates.

Converts per-stimulus (ty, tz, s) rendering parameters to pixel-space
(cx_px, cy_px, r_px) for building per-stimulus guidance masks.

Camera setup (fixed across all HVM stimuli):
  - Perspective camera at z=10, target at origin, FOV=45 degrees
  - ty = horizontal translation (positive = right in image)
  - tz = vertical translation (positive = up in image, i.e. smaller y)
  - s  = object scale (scales the apparent object radius)
  - px_per_unit includes a 1.64× empirical correction vs. the geometric formula
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

from config_const import HVM_METADATA_PATH, HVM_BBOXES_PATH, HVM_RADIUS_FRAC


@lru_cache(maxsize=1)
def load_hvm_metadata() -> list[dict]:
    """Load and cache hvm10_metadata.json (450 records, one per stimulus)."""
    return json.loads(HVM_METADATA_PATH.read_text())


@lru_cache(maxsize=1)
def load_hvm_bboxes() -> list[dict]:
    """Load cached per-stimulus bounding boxes from hvm10_bboxes.json.

    Each record: {idx, cx, cy, half, source, score}
    cx/cy/half are in 276×276 pixel space.
    source is 'gdino' or 'projected'.
    """
    return json.loads(HVM_BBOXES_PATH.read_text())


def hvm_object_pixel_coords(
    ty: float,
    tz: float,
    s: float,
    image_size: int = 512,
    fov_deg: float = 45.0,
    cam_z: float = 10.0,
    base_radius_frac: float = HVM_RADIUS_FRAC,
) -> tuple[float, float, float]:
    """Project HVM rendering params to pixel-space object footprint.

    Args:
        ty:               Horizontal translation in world units (right = positive).
        tz:               Vertical translation in world units (up = positive).
        s:                Object scale factor.
        image_size:       Square canvas side length in pixels.
        fov_deg:          Camera field of view in degrees (full horizontal).
        cam_z:            Camera z-position (object is at origin).
        base_radius_frac: Aperture radius as fraction of image_size at s=1.

    Returns:
        (cx_px, cy_px, r_px): Object center and radius in pixel coordinates.
        cy_px is measured top-down (standard image convention).
    """
    half_w = math.tan(math.radians(fov_deg / 2)) * cam_z
    px_per_unit = (image_size / 2) / half_w * 2.5  # 2.5× empirical correction (visual calibration)
    cx = image_size / 2 + ty * px_per_unit   # ty+ = right in image
    cy = image_size / 2 - tz * px_per_unit   # tz+ = up in 3D = smaller y in image
    r = image_size * base_radius_frac * s
    return cx, cy, r

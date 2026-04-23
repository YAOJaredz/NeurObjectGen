"""Build per-stimulus metadata JSON for HVM10 stimuli.

For each of the 450 stimuli (10 categories × 45 variations) writes one entry
into cache/hvm10_metadata.json with transform params parsed from the filename
plus fixed scene properties from the category's scene JSON.

Output structure (list of 450 dicts, ordered apple×45 … turtle×45):
  {
    "idx":       int,          # global index 0-449
    "category":  str,
    "filename":  str,          # original engram PNG filename
    "ty":        float,        # y-translation
    "tz":        float,        # z-translation
    "rxy":       float,        # rotation xy (degrees)
    "rxz":       float,        # rotation xz (degrees)
    "ryz":       float,        # rotation yz (degrees)
    "s":         float,        # scale
    "camera":    dict,         # camera intrinsics/extrinsics
    "lights":    dict,         # lighting setup
    "material":  dict,         # object material properties
    "duration_ms": int
  }
"""

import json
import os
import re
import sys

sys.path.append('.')

import numpy as np
from pathlib import Path

from config_const import CACHE_DIR, HVM_CATEGORIES
from HexPred.constants import ENGRAM_PATH

ENGRAM_PATH = Path(ENGRAM_PATH)
HVM10_FILENAMES_DIR = ENGRAM_PATH / "users" / "Younah" / "mkTurkdemo_imagefiles" / "hvm10_filenames"
HVM10_SCENE_DIR     = ENGRAM_PATH / "Data" / "West" / "Saved_Images" / "E8"
OUT_PATH            = CACHE_DIR / "hvm10_metadata.json"


def parse_filename_params(filename: str) -> dict:
    params = {}
    for key in ("ty", "tz", "rxy", "rxz", "ryz", "s"):
        match = re.search(key + r'([-+]?[0-9]*\.?[0-9]+)', filename)
        params[key] = float(match.group(1))
    return params


def load_scene_constants(scene: dict) -> dict:
    cam = scene["CAMERAS"]["camera00"]
    obj_key = list(scene["OBJECTS"].keys())[0]
    obj = scene["OBJECTS"][obj_key]
    return {
        "camera": {
            "type":        cam["type"],
            "fieldOfView": cam["fieldOfView"],
            "near":        cam["near"],
            "far":         cam["far"],
            "position":    {k: v[0] for k, v in cam["position"].items()},
            "target":      {k: v[0] for k, v in cam["targetTHREEJS"].items()},
        },
        "lights": {
            name: {
                "type":      light["type"],
                "color":     light["color"],
                "intensity": light["intensity"][0],
                "position":  {k: v[0] for k, v in light["position"].items()},
            }
            for name, light in scene["LIGHTS"].items()
        },
        "material": obj["material"],
        "duration_ms": scene["durationMS"][0],
    }


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    global_idx = 0

    for cat in HVM_CATEGORIES:
        all_files = np.sort(os.listdir(HVM10_FILENAMES_DIR / cat))
        all_files = all_files[all_files != "Thumbs.db"]

        scene_name = f"hvm10_{cat}_45_20240906"
        scene = json.load(open(HVM10_SCENE_DIR / scene_name / f"{scene_name}.json"))
        imageidx = scene["IMAGES"]["imageidx"]
        scene_consts = load_scene_constants(scene)

        selected = all_files[imageidx]
        pngs = [f for f in selected if f.endswith(".png")]
        assert len(pngs) == 45, f"{cat}: expected 45, got {len(pngs)}"

        for filename in pngs:
            record = {
                "idx":      global_idx,
                "category": cat,
                "filename": filename,
                **parse_filename_params(filename),
                **scene_consts,
            }
            records.append(record)
            global_idx += 1

    OUT_PATH.write_text(json.dumps(records, indent=2))
    print(f"saved {len(records)} records -> {OUT_PATH}")


if __name__ == "__main__":
    main()

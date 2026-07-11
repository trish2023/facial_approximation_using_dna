#!/usr/bin/env python3
"""Recreate the template landmark output used by downstream face stages.

The project already stores a validated landmark layout in
`outputs/template_landmarks_debug.json`. This stage materializes the
publication-facing artifact expected by the pipeline:

- `data/processed/template_landmarks.json`
- `outputs/template_landmarks_overlay.png`

If a future template-specific detector is added, this wrapper can be swapped
out without changing downstream stages.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TEMPLATE_SELECTION_JSON = OUTPUT_DIR / "template_selection.json"
TEMPLATE_DEBUG_JSON = OUTPUT_DIR / "template_landmarks_debug.json"
DEFAULT_OUTPUT_JSON = PROCESSED_DIR / "template_landmarks.json"
DEFAULT_OVERLAY = OUTPUT_DIR / "template_landmarks_overlay.png"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)s]  %(message)s")
LOG = logging.getLogger("landmark_detection")


def load_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def load_template_image(template_path: str | None) -> Image.Image:
    if not template_path:
        raise FileNotFoundError("template_path missing from template selection JSON.")
    resolved = Path(template_path)
    if not resolved.is_absolute():
        resolved = (PROJECT_ROOT / resolved).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Template image not found: {resolved}")
    return Image.open(resolved).convert("RGB")


def build_landmark_payload(template_debug: dict[str, Any]) -> dict[str, Any]:
    original = template_debug.get("original_68_landmarks")
    boundary = template_debug.get("boundary_landmarks")
    if not isinstance(original, list) or not original:
        raise ValueError("template_landmarks_debug.json must contain original_68_landmarks.")
    if not isinstance(boundary, list):
        boundary = []

    landmarks: list[dict[str, float]] = []
    for entry in original + boundary:
        if not isinstance(entry, dict):
            continue
        landmarks.append(
            {
                "id": int(entry["id"]),
                "x": float(entry["x"]),
                "y": float(entry["y"]),
            }
        )

    landmarks.sort(key=lambda item: item["id"])
    return {
        "template_name": template_debug.get("template_name"),
        "landmark_count": len(landmarks),
        "image_width": template_debug.get("image_width"),
        "image_height": template_debug.get("image_height"),
        "landmarks": landmarks,
    }


def create_overlay(template_image: Image.Image, landmarks: list[dict[str, float]], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(template_image)
    xs = [point["x"] for point in landmarks]
    ys = [point["y"] for point in landmarks]
    ax.scatter(xs[:68], ys[:68], s=12, c="#2563eb", label="Facial landmarks")
    if len(landmarks) > 68:
        ax.scatter(xs[68:], ys[68:], s=22, c="#f97316", label="Boundary anchors")
    ax.set_title("Template landmark overlay", fontweight="bold")
    ax.axis("off")
    ax.legend(loc="lower right", frameon=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize template landmark outputs for the pipeline.")
    parser.add_argument("--template-selection", default=str(TEMPLATE_SELECTION_JSON))
    parser.add_argument("--template-debug", default=str(TEMPLATE_DEBUG_JSON))
    parser.add_argument("--output-json", default=str(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--overlay-output", default=str(DEFAULT_OVERLAY))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    template_selection = load_json_object(Path(args.template_selection))
    template_debug = load_json_object(Path(args.template_debug))
    if "landmarks" in template_debug:
        payload = template_debug
    else:
        payload = build_landmark_payload(template_debug)

    output_json = write_json(Path(args.output_json), payload)
    template_image = load_template_image(template_selection.get("template_path"))
    overlay = create_overlay(template_image, payload["landmarks"], Path(args.overlay_output))

    LOG.info("Saved template landmarks to %s", output_json)
    LOG.info("Saved landmark overlay to %s", overlay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

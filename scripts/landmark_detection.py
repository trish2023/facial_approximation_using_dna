#!/usr/bin/env python3
"""
Detect facial landmarks on the selected template image for Phase 6 warping.

Landmark detection is a critical prerequisite for morphology-driven warping:
the pipeline needs stable anchor points before any trait-based deformation can
be applied. Boundary points are added beyond the canonical 68 landmarks so a
later Delaunay triangulation can cover the full image extent rather than only
the central facial area, which helps prevent edge tearing and unstable warps.
"""

from __future__ import annotations

import argparse
import bz2
import json
import logging
from pathlib import Path
from typing import Any

import cv2
import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DLIB_DIR = PROJECT_ROOT / "data" / "reference" / "dlib"
DEFAULT_TEMPLATE_SELECTION = OUTPUT_DIR / "template_selection.json"
DEFAULT_LANDMARK_JSON = PROCESSED_DIR / "template_landmarks.json"
DEFAULT_DEBUG_JSON = OUTPUT_DIR / "template_landmarks_debug.json"
DEFAULT_OVERLAY = OUTPUT_DIR / "template_landmarks_overlay.png"
PREDICTOR_BZ2_URL = "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
PREDICTOR_BZ2_PATH = DLIB_DIR / "shape_predictor_68_face_landmarks.dat.bz2"
PREDICTOR_DAT_PATH = DLIB_DIR / "shape_predictor_68_face_landmarks.dat"

REGION_MAPPING = {
    "jaw": list(range(0, 17)),
    "left_eyebrow": list(range(17, 22)),
    "right_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),
    "left_eye": list(range(36, 42)),
    "right_eye": list(range(42, 48)),
    "mouth": list(range(48, 68)),
    "boundary": list(range(68, 76)),
}

REGION_COLORS = {
    "jaw": (255, 0, 0),
    "left_eyebrow": (0, 128, 255),
    "right_eyebrow": (0, 180, 255),
    "nose": (0, 255, 0),
    "left_eye": (255, 255, 0),
    "right_eye": (255, 200, 0),
    "mouth": (255, 0, 255),
    "boundary": (180, 180, 180),
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("landmark_detection")


def import_dlib():
    """Import dlib lazily so the module can still be compiled without it."""
    try:
        import dlib  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "dlib is required for landmark detection but is not installed in this environment."
        ) from exc
    return dlib


def ensure_shape_predictor(
    predictor_url: str = PREDICTOR_BZ2_URL,
    bz2_path: Path = PREDICTOR_BZ2_PATH,
    dat_path: Path = PREDICTOR_DAT_PATH,
) -> Path:
    """
    Ensure the 68-point dlib predictor exists locally.

    The predictor is downloaded only once, then decompressed from .bz2 to the
    .dat file dlib expects.
    """
    dat_path.parent.mkdir(parents=True, exist_ok=True)

    if dat_path.exists():
        LOG.info("Using existing dlib predictor at %s", dat_path)
        return dat_path

    if not bz2_path.exists():
        LOG.info("Downloading dlib predictor to %s", bz2_path)
        response = requests.get(predictor_url, stream=True, timeout=120)
        response.raise_for_status()
        with bz2_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    LOG.info("Decompressing %s to %s", bz2_path, dat_path)
    with bz2.BZ2File(bz2_path, "rb") as source, dat_path.open("wb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)

    if not dat_path.exists():
        raise FileNotFoundError(
            f"Failed to prepare dlib predictor at {dat_path} after download/decompression."
        )
    return dat_path


def load_template_selection(json_path: str | Path) -> dict[str, Any]:
    """Load outputs/template_selection.json and return its payload."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Template selection JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Template selection JSON must contain an object: {path}")
    return payload


def resolve_template_path(template_path: str | Path) -> Path:
    """Resolve a template path relative to the project root when needed."""
    path = Path(template_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_template_image(template_path: str | Path) -> tuple[Path, Any]:
    """Load the selected template image with OpenCV and validate it."""
    path = resolve_template_path(template_path)
    if not path.exists():
        raise FileNotFoundError(f"Template image not found: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"OpenCV could not load template image: {path}")
    return path, image


def detect_single_face(detector, image_gray, template_path: Path):
    """Run dlib face detection and require exactly one face."""
    faces = detector(image_gray)
    if len(faces) == 0:
        raise ValueError(
            f"No face detected in template image {template_path}. "
            "The template is unsuitable; exactly one clear frontal face is required."
        )
    if len(faces) > 1:
        raise ValueError(
            f"Multiple faces detected in template image {template_path}. "
            "Only one face should be present in a valid template."
        )
    return faces[0]


def extract_68_landmarks(shape) -> list[dict[str, int]]:
    """Convert dlib's shape object into serialisable landmark dictionaries."""
    return [
        {"id": idx, "x": int(shape.part(idx).x), "y": int(shape.part(idx).y)}
        for idx in range(68)
    ]


def add_boundary_points(image_width: int, image_height: int) -> list[dict[str, int]]:
    """
    Add 8 boundary points around the image edges.

    These are needed so later Delaunay triangulation can span the full template
    canvas and keep edge triangles well constrained during facial warping.
    """
    max_x = image_width - 1
    max_y = image_height - 1
    mid_x = image_width // 2
    mid_y = image_height // 2

    coords = [
        (0, 0),
        (mid_x, 0),
        (max_x, 0),
        (0, mid_y),
        (max_x, mid_y),
        (0, max_y),
        (mid_x, max_y),
        (max_x, max_y),
    ]
    return [
        {"id": 68 + idx, "x": int(x), "y": int(y)}
        for idx, (x, y) in enumerate(coords)
    ]


def get_landmark_region(landmark_id: int) -> str:
    """Map a landmark ID to its anatomical region name."""
    for region, ids in REGION_MAPPING.items():
        if landmark_id in ids:
            return region
    raise ValueError(f"Unknown landmark id outside defined regions: {landmark_id}")


def draw_legend(image, legend_items: list[tuple[str, tuple[int, int, int]]]) -> None:
    """Draw a compact legend for the overlay image."""
    x0 = 15
    y0 = 20
    line_h = 24
    box_w = 230
    box_h = line_h * len(legend_items) + 15
    cv2.rectangle(image, (x0 - 10, y0 - 15), (x0 + box_w, y0 + box_h), (255, 255, 255), -1)
    cv2.rectangle(image, (x0 - 10, y0 - 15), (x0 + box_w, y0 + box_h), (0, 0, 0), 1)

    for idx, (label, color) in enumerate(legend_items):
        y = y0 + idx * line_h
        cv2.circle(image, (x0 + 8, y), 6, color, -1)
        cv2.putText(
            image,
            label,
            (x0 + 22, y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )


def create_overlay_image(image, landmarks: list[dict[str, int]], output_path: str | Path) -> Path:
    """Render a landmark overlay with IDs and a color legend."""
    overlay = image.copy()
    for landmark in landmarks:
        region = get_landmark_region(int(landmark["id"]))
        color = REGION_COLORS[region]
        x = int(landmark["x"])
        y = int(landmark["y"])
        cv2.circle(overlay, (x, y), 3, color, -1)
        cv2.putText(
            overlay,
            str(landmark["id"]),
            (x + 4, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )

    draw_legend(
        overlay,
        [
            ("Jaw", REGION_COLORS["jaw"]),
            ("Left eyebrow", REGION_COLORS["left_eyebrow"]),
            ("Right eyebrow", REGION_COLORS["right_eyebrow"]),
            ("Nose", REGION_COLORS["nose"]),
            ("Left eye", REGION_COLORS["left_eye"]),
            ("Right eye", REGION_COLORS["right_eye"]),
            ("Mouth", REGION_COLORS["mouth"]),
            ("Boundary", REGION_COLORS["boundary"]),
        ],
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), overlay):
        raise ValueError(f"Failed to write overlay image to {output}")
    return output


def write_json(path: str | Path, payload: Any) -> Path:
    """Write a JSON payload to disk."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output


def detect_template_landmarks(
    template_selection_path: str | Path = DEFAULT_TEMPLATE_SELECTION,
    landmarks_output_path: str | Path = DEFAULT_LANDMARK_JSON,
    debug_output_path: str | Path = DEFAULT_DEBUG_JSON,
    overlay_output_path: str | Path = DEFAULT_OVERLAY,
) -> dict[str, Any]:
    """Run the full template landmark detection workflow."""
    selection = load_template_selection(template_selection_path)
    template_name = selection.get("template_name", "unknown_template")
    template_path_value = selection.get("template_path")
    if not template_path_value:
        raise ValueError(
            f"Template selection file {template_selection_path} does not contain template_path."
        )

    predictor_path = ensure_shape_predictor()
    dlib = import_dlib()
    template_path, image = load_template_image(template_path_value)
    image_height, image_width = image.shape[:2]
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(predictor_path))
    face_rect = detect_single_face(detector, image_gray, template_path)
    shape = predictor(image_gray, face_rect)

    original_landmarks = extract_68_landmarks(shape)
    boundary_landmarks = add_boundary_points(image_width=image_width, image_height=image_height)
    all_landmarks = original_landmarks + boundary_landmarks

    landmark_payload = {
        "template_name": template_name,
        "image_width": image_width,
        "image_height": image_height,
        "landmark_count": len(all_landmarks),
        "landmarks": all_landmarks,
        "region_mapping": REGION_MAPPING,
    }
    debug_payload = {
        "template_name": template_name,
        "image_width": image_width,
        "image_height": image_height,
        "original_68_landmarks": original_landmarks,
        "boundary_landmarks": boundary_landmarks,
    }

    write_json(landmarks_output_path, landmark_payload)
    write_json(debug_output_path, debug_payload)
    create_overlay_image(image, all_landmarks, overlay_output_path)

    LOG.info("Selected template: %s", template_path)
    LOG.info("Image size: %dx%d", image_width, image_height)
    LOG.info("Face detection success: exactly one face found")
    LOG.info("68 landmarks detected")
    LOG.info("8 boundary points added")
    LOG.info("Total landmarks = %d", len(all_landmarks))

    return landmark_payload


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect 68-point + boundary landmarks on the selected template image."
    )
    parser.add_argument(
        "--template-selection",
        default=str(DEFAULT_TEMPLATE_SELECTION),
        help="Path to outputs/template_selection.json (default: %(default)s).",
    )
    parser.add_argument(
        "--landmarks-output",
        default=str(DEFAULT_LANDMARK_JSON),
        help="Path to data/processed/template_landmarks.json (default: %(default)s).",
    )
    parser.add_argument(
        "--debug-output",
        default=str(DEFAULT_DEBUG_JSON),
        help="Path to outputs/template_landmarks_debug.json (default: %(default)s).",
    )
    parser.add_argument(
        "--overlay-output",
        default=str(DEFAULT_OVERLAY),
        help="Path to outputs/template_landmarks_overlay.png (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    detect_template_landmarks(
        template_selection_path=args.template_selection,
        landmarks_output_path=args.landmarks_output,
        debug_output_path=args.debug_output,
        overlay_output_path=args.overlay_output,
    )


if __name__ == "__main__":
    main()

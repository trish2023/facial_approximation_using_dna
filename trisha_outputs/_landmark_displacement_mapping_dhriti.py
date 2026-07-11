#!/usr/bin/env python3
"""
Map currently available PRS-derived facial morphology traits to landmark displacements.

Only jawline and chin landmarks are modified at this stage because the current
GWAS-backed morphology traits in the pipeline cover chin and lower-cheek
variation only. No validated traits are yet available for the eyes, nose,
mouth, forehead, or eyebrows, so this script intentionally leaves those facial
regions unchanged. As future facial GWAS releases add validated signals for
additional regions, the trait-to-landmark mapping can be extended without
changing the downstream warping interface.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MORPHOLOGY_JSON = OUTPUT_DIR / "morphology_traits.json"
FALLBACK_MORPHOLOGY_JSON = PROCESSED_DIR / "morphology_traits.json"
DEFAULT_LANDMARKS_JSON = PROCESSED_DIR / "template_landmarks.json"
DEFAULT_DISPLACEMENTS_JSON = PROCESSED_DIR / "landmark_displacements.json"
DEFAULT_SUMMARY_JSON = OUTPUT_DIR / "displacement_summary.json"
DEFAULT_OVERLAY_PATH = OUTPUT_DIR / "landmark_displacements_overlay.png"
SCALE_FACTOR = 5.0
MAX_DISPLACEMENT_PIXELS = 20.0

REGION_LANDMARKS = {
    "Forehead": [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
    "Eyes": [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
    "Uppernose": [27, 28, 29],
    "Lowernose": [30, 31, 32, 33, 34, 35],
    "Uppercheek": [1, 2, 14, 15],
    "Lowercheek": [3, 4, 5, 11, 12, 13],
    "Mouth": [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67],
    "Chin": [6, 7, 8, 9, 10],
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("landmark_displacement_mapping")


@dataclass
class TraitRecord:
    key: str
    canonical_trait: str
    z_score: float
    direction: str
    magnitude: str
    confidence_score: float
    displacement_pixels: float
    landmarks: list[int]


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


def resolve_morphology_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path).resolve()
    if DEFAULT_MORPHOLOGY_JSON.exists():
        return DEFAULT_MORPHOLOGY_JSON
    if FALLBACK_MORPHOLOGY_JSON.exists():
        LOG.warning(
            "Default morphology file %s not found; falling back to %s.",
            DEFAULT_MORPHOLOGY_JSON,
            FALLBACK_MORPHOLOGY_JSON,
        )
        return FALLBACK_MORPHOLOGY_JSON
    raise FileNotFoundError(
        f"Neither {DEFAULT_MORPHOLOGY_JSON} nor fallback {FALLBACK_MORPHOLOGY_JSON} exists."
    )


def coerce_float(value: Any, field_name: str, trait_key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Trait {trait_key} has non-numeric {field_name}: {value!r}") from exc
    if math.isnan(number) or math.isinf(number):
        raise ValueError(f"Trait {trait_key} has invalid {field_name}: {value!r}")
    return number


def find_canonical_trait_name(trait_key: str) -> str | None:
    parts = trait_key.split('_')
    if len(parts) >= 2:
        regions_part = parts[1]
        regions = regions_part.split('-')
        valid_regions = [r for r in regions if r in REGION_LANDMARKS]
        if valid_regions:
            return "-".join(valid_regions)
    return None


def load_landmarks(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, float]]]:
    payload = load_json_object(path)
    landmarks = payload.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError(f"Landmarks payload in {path} must contain a 'landmarks' list.")

    indexed: dict[int, dict[str, float]] = {}
    for item in landmarks:
        if not isinstance(item, dict):
            raise ValueError(f"Each landmark entry in {path} must be an object.")
        landmark_id = int(item["id"])
        indexed[landmark_id] = {
            "x": float(item["x"]),
            "y": float(item["y"]),
        }
    return payload, indexed


def load_trait_records(path: Path) -> tuple[list[TraitRecord], int]:
    payload = load_json_object(path)
    loaded_trait_count = len(payload)
    mapped_traits: list[TraitRecord] = []

    for trait_key, raw_record in payload.items():
        canonical_trait = find_canonical_trait_name(trait_key)
        if canonical_trait is None:
            continue
        if not isinstance(raw_record, dict):
            raise ValueError(f"Trait {trait_key} must map to an object.")

        z_score = coerce_float(raw_record.get("z_score"), "z_score", trait_key)
        confidence_score = coerce_float(
            raw_record.get("confidence_score"), "confidence_score", trait_key
        )
        direction = str(raw_record.get("direction", "Unavailable"))
        magnitude = str(raw_record.get("magnitude", "Unavailable"))

        if confidence_score <= 0 or z_score == 0:
            continue

        displacement_pixels = max(
            -MAX_DISPLACEMENT_PIXELS,
            min(MAX_DISPLACEMENT_PIXELS, z_score * SCALE_FACTOR),
        )

        landmarks = []
        for r in canonical_trait.split('-'):
            landmarks.extend(REGION_LANDMARKS[r])
        landmarks = list(set(landmarks))

        mapped_traits.append(
            TraitRecord(
                key=trait_key,
                canonical_trait=canonical_trait,
                z_score=z_score,
                direction=direction,
                magnitude=magnitude,
                confidence_score=confidence_score,
                displacement_pixels=displacement_pixels,
                landmarks=landmarks,
            )
        )

    return mapped_traits, loaded_trait_count


def apply_trait_displacement(
    trait: TraitRecord,
    accumulation_map: dict[int, dict[str, float]],
    landmarks: dict[int, dict[str, float]],
    cx: float,
    cy: float,
) -> None:
    weight = trait.confidence_score if trait.confidence_score > 0 else 1.0
    
    for landmark_id in trait.landmarks:
        lm = landmarks[landmark_id]
        dx = lm["x"] - cx
        dy = lm["y"] - cy
        dist = math.hypot(dx, dy)
        
        if dist > 0:
            nx, ny = dx / dist, dy / dist
            
            dx_contrib = nx * trait.displacement_pixels
            dy_contrib = ny * trait.displacement_pixels
            
            accumulation_map[landmark_id]["dx_sum"] += dx_contrib * weight
            accumulation_map[landmark_id]["dy_sum"] += dy_contrib * weight
            
            accumulation_map[landmark_id]["dx_sq_sum"] += (dx_contrib ** 2) * weight
            accumulation_map[landmark_id]["dy_sq_sum"] += (dy_contrib ** 2) * weight
            
            accumulation_map[landmark_id]["weight_x"] += weight
            accumulation_map[landmark_id]["weight_y"] += weight


def build_displacements(
    landmarks: dict[int, dict[str, float]],
    traits: list[TraitRecord],
) -> dict[int, dict[str, float]]:
    accumulation_map: dict[int, dict[str, float]] = {
        landmark_id: {
            "dx_sum": 0.0, "dy_sum": 0.0,
            "dx_sq_sum": 0.0, "dy_sq_sum": 0.0,
            "weight_x": 0.0, "weight_y": 0.0
        }
        for landmark_id in landmarks
    }
    
    if not landmarks:
        return {}
        
    cx = sum(lm["x"] for lm in landmarks.values()) / len(landmarks)
    cy = sum(lm["y"] for lm in landmarks.values()) / len(landmarks)
    
    for trait in traits:
        apply_trait_displacement(trait, accumulation_map, landmarks, cx, cy)
        
    displacement_map: dict[int, dict[str, float]] = {}
    for landmark_id, acc in accumulation_map.items():
        w_x = acc["weight_x"]
        w_y = acc["weight_y"]
        
        if w_x > 0:
            rms_x = math.sqrt(acc["dx_sq_sum"] / w_x)
            dx = math.copysign(rms_x, acc["dx_sum"])
        else:
            dx = 0.0
            
        if w_y > 0:
            rms_y = math.sqrt(acc["dy_sq_sum"] / w_y)
            dy = math.copysign(rms_y, acc["dy_sum"])
        else:
            dy = 0.0
            
        displacement_map[landmark_id] = {"dx": dx, "dy": dy}
        
    return displacement_map


def serialise_displacements(
    displacement_map: dict[int, dict[str, float]]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for landmark_id in sorted(displacement_map):
        dx = displacement_map[landmark_id]["dx"]
        dy = displacement_map[landmark_id]["dy"]
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        output[f"landmark_{landmark_id}"] = {
            "dx": round(dx, 4),
            "dy": round(dy, 4),
        }
    return output


def displacement_magnitude(vector: dict[str, float]) -> float:
    return math.hypot(vector["dx"], vector["dy"])


def build_summary(
    traits: list[TraitRecord],
    displacement_map: dict[int, dict[str, float]],
) -> dict[str, Any]:
    non_zero_landmarks = {
        landmark_id: vector
        for landmark_id, vector in displacement_map.items()
        if abs(vector["dx"]) > 1e-9 or abs(vector["dy"]) > 1e-9
    }
    magnitudes = [displacement_magnitude(vector) for vector in non_zero_landmarks.values()]

    return {
        "traits_used": [
            {
                "trait": trait.key,
                "canonical_trait": trait.canonical_trait,
                "z_score": round(trait.z_score, 4),
                "direction": trait.direction,
                "magnitude": trait.magnitude,
                "confidence_score": round(trait.confidence_score, 4),
                "displacement_pixels": round(trait.displacement_pixels, 4),
                "landmarks": trait.landmarks,
            }
            for trait in traits
        ],
        "landmarks_modified": sorted(non_zero_landmarks),
        "max_displacement": round(max(magnitudes, default=0.0), 4),
        "mean_displacement": round(mean(magnitudes) if magnitudes else 0.0, 4),
        "notes": [
            "Only jawline and chin landmarks are modified because the current validated GWAS-derived morphology traits in this pipeline cover chin and lower-cheek variation only.",
            "This displacement mapping reflects the currently available morphology traits and is designed to remain conservative outside the lower facial contour.",
            "Future facial GWAS releases can extend this mapping to eyes, nose, mouth, forehead, and eyebrows once validated regional traits become available.",
        ],
    }


def create_overlay(
    landmarks_payload: dict[str, Any],
    landmarks: dict[int, dict[str, float]],
    displacement_map: dict[int, dict[str, float]],
    output_path: Path,
) -> Path:
    image_width = int(landmarks_payload.get("image_width", 1000))
    image_height = int(landmarks_payload.get("image_height", 1000))
    header_height = 90
    canvas = np.full((image_height + header_height, image_width, 3), 255, dtype=np.uint8)

    cv2.putText(
        canvas,
        "Landmark Displacement Mapping",
        (24, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    legend_items = [
        ("Original landmarks", (255, 0, 0)),
        ("Displaced landmarks", (0, 0, 255)),
        ("Displacement arrows", (120, 120, 120)),
    ]
    legend_x = 24
    legend_y = 62
    for label, color in legend_items:
        cv2.circle(canvas, (legend_x, legend_y), 6, color, -1)
        cv2.putText(
            canvas,
            label,
            (legend_x + 16, legend_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )
        legend_x += 210

    cv2.rectangle(
        canvas,
        (0, header_height),
        (image_width - 1, image_height + header_height - 1),
        (230, 230, 230),
        1,
    )

    for landmark_id in sorted(landmarks):
        start_x = int(round(landmarks[landmark_id]["x"]))
        start_y = int(round(landmarks[landmark_id]["y"])) + header_height
        end_x = int(round(landmarks[landmark_id]["x"] + displacement_map[landmark_id]["dx"]))
        end_y = int(round(landmarks[landmark_id]["y"] + displacement_map[landmark_id]["dy"])) + header_height

        cv2.arrowedLine(
            canvas,
            (start_x, start_y),
            (end_x, end_y),
            (120, 120, 120),
            1,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.circle(canvas, (start_x, start_y), 3, (255, 0, 0), -1)
        cv2.circle(canvas, (end_x, end_y), 3, (0, 0, 255), -1)
        cv2.putText(
            canvas,
            str(landmark_id),
            (start_x + 4, start_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (90, 40, 20),
            1,
            cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise ValueError(f"Failed to write overlay image to {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Map current morphology traits to jawline/chin landmark displacements."
    )
    parser.add_argument(
        "--morphology",
        default=None,
        help=(
            "Path to morphology_traits.json. Defaults to outputs/morphology_traits.json "
            "with fallback to data/processed/morphology_traits.json."
        ),
    )
    parser.add_argument(
        "--landmarks",
        default=str(DEFAULT_LANDMARKS_JSON),
        help="Path to data/processed/template_landmarks.json (default: %(default)s).",
    )
    parser.add_argument(
        "--displacements-output",
        default=str(DEFAULT_DISPLACEMENTS_JSON),
        help="Path to data/processed/landmark_displacements.json (default: %(default)s).",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_JSON),
        help="Path to outputs/displacement_summary.json (default: %(default)s).",
    )
    parser.add_argument(
        "--overlay-output",
        default=str(DEFAULT_OVERLAY_PATH),
        help="Path to outputs/landmark_displacements_overlay.png (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    morphology_path = resolve_morphology_path(args.morphology)
    landmarks_path = Path(args.landmarks).resolve()
    displacements_output_path = Path(args.displacements_output).resolve()
    summary_output_path = Path(args.summary_output).resolve()
    overlay_output_path = Path(args.overlay_output).resolve()

    landmarks_payload, landmarks = load_landmarks(landmarks_path)
    traits, loaded_trait_count = load_trait_records(morphology_path)

    displacement_map = build_displacements(landmarks, traits)
    displacement_payload = serialise_displacements(displacement_map)
    summary_payload = build_summary(traits, displacement_map)

    write_json(displacements_output_path, displacement_payload)
    write_json(summary_output_path, summary_payload)
    create_overlay(landmarks_payload, landmarks, displacement_map, overlay_output_path)

    modified_landmarks = summary_payload["landmarks_modified"]
    max_displacement = summary_payload["max_displacement"]

    LOG.info("Saved landmark displacements to %s", displacements_output_path)
    LOG.info("Saved displacement summary to %s", summary_output_path)
    LOG.info("Saved displacement overlay to %s", overlay_output_path)

    print(f"Number of morphology traits loaded: {loaded_trait_count}")
    print(f"Traits contributing displacement: {[trait.key for trait in traits]}")
    print(f"Landmarks modified: {modified_landmarks}")
    print(f"Maximum displacement magnitude: {max_displacement:.4f} pixels")
    print(
        "Documentation: only jaw/chin landmarks are modified because current validated "
        "GWAS-derived morphology traits in this pipeline cover chin and lower-cheek variation only."
    )
    print(
        "Documentation: the current displacement mapping reflects the available GWAS traits "
        "and intentionally leaves eyes, nose, mouth, and forehead unchanged."
    )
    print(
        "Documentation: future facial GWAS releases can extend this mapping to eyes, nose, "
        "mouth, forehead, and eyebrows once validated regional traits become available."
    )


if __name__ == "__main__":
    main()

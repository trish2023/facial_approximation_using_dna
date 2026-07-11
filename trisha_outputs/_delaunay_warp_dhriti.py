#!/usr/bin/env python3
"""
Phase 7: piecewise affine facial warping using Delaunay triangulation.

This script consumes the selected face template, the original landmark layout,
and a landmark displacement map to produce a morphology-warped face plus QC
artifacts for downstream reporting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import Delaunay


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference" / "face_templates"

DEFAULT_TEMPLATE_SELECTION = OUTPUT_DIR / "template_selection.json"
DEFAULT_TEMPLATE = None
DEFAULT_LANDMARKS = PROCESSED_DIR / "template_landmarks.json"
DEFAULT_DISPLACEMENTS = OUTPUT_DIR / "displacement_map.json"
FALLBACK_DISPLACEMENTS = PROCESSED_DIR / "landmark_displacements.json"
DEFAULT_MORPHOLOGY = OUTPUT_DIR / "morphology_traits.json"
FALLBACK_MORPHOLOGY = PROCESSED_DIR / "morphology_traits.json"

DEFAULT_WARPED_FACE = OUTPUT_DIR / "morphology_warped_face.png"
DEFAULT_HEATMAP = OUTPUT_DIR / "warp_difference_heatmap.png"
DEFAULT_COMPARISON = OUTPUT_DIR / "warp_comparison.png"
DEFAULT_QC_JSON = OUTPUT_DIR / "warp_qc.json"

BOUNDARY_IDS = tuple(range(68, 76))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("delaunay_warp")


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


def resolve_template_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    selection = load_json_object(DEFAULT_TEMPLATE_SELECTION)
    template_value = selection.get("template_path")
    if not template_value:
        raise ValueError(
            f"Template selection file {DEFAULT_TEMPLATE_SELECTION} does not contain template_path."
        )
    template_path = Path(str(template_value))
    return template_path if template_path.is_absolute() else (PROJECT_ROOT / template_path).resolve()


def resolve_optional_input(explicit_path: str | None, default_path: Path, fallback_path: Path) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()
    if default_path.exists():
        return default_path
    if fallback_path.exists():
        LOG.warning("Default input %s not found; falling back to %s.", default_path, fallback_path)
        return fallback_path
    raise FileNotFoundError(f"Neither {default_path} nor fallback {fallback_path} exists.")


def load_template_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not load template image: {path}")
    return image


def load_landmark_points(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    payload = load_json_object(path)
    landmarks = payload.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError(f"Landmark JSON {path} must contain a 'landmarks' list.")

    points: list[tuple[float, float]] = []
    ids: list[int] = []
    for item in landmarks:
        if not isinstance(item, dict):
            raise ValueError(f"Each landmark entry in {path} must be an object.")
        ids.append(int(item["id"]))
        points.append((float(item["x"]), float(item["y"])))

    expected_ids = list(range(len(points)))
    if ids != expected_ids:
        raise ValueError(f"Landmark IDs in {path} must be sequential starting from 0; found {ids[:10]}...")

    return payload, np.asarray(points, dtype=np.float32)


def load_displacement_payload(path: Path) -> dict[str, Any]:
    payload = load_json_object(path)
    if not payload:
        raise ValueError(f"Displacement JSON {path} is empty.")
    return payload


def build_target_points(original_points: np.ndarray, displacement_payload: dict[str, Any]) -> np.ndarray:
    landmark_count = len(original_points)
    target_points = np.copy(original_points)

    for landmark_id in range(landmark_count):
        key = f"landmark_{landmark_id}"
        entry = displacement_payload.get(key)
        if entry is None:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{key} entry in displacement map must be an object.")

        if {"target_x", "target_y"} <= set(entry):
            target_points[landmark_id] = np.asarray(
                [float(entry["target_x"]), float(entry["target_y"])],
                dtype=np.float32,
            )
            continue
        if {"displaced_x", "displaced_y"} <= set(entry):
            target_points[landmark_id] = np.asarray(
                [float(entry["displaced_x"]), float(entry["displaced_y"])],
                dtype=np.float32,
            )
            continue
        if {"x", "y"} <= set(entry) and {"original_x", "original_y"} <= set(entry):
            target_points[landmark_id] = np.asarray([float(entry["x"]), float(entry["y"])], dtype=np.float32)
            continue
        if {"dx", "dy"} <= set(entry):
            target_points[landmark_id] = original_points[landmark_id] + np.asarray(
                [float(entry["dx"]), float(entry["dy"])],
                dtype=np.float32,
            )
            continue
        raise ValueError(
            f"{key} entry in displacement map must contain either dx/dy or explicit displaced coordinates."
        )

    # Keep the 8 boundary anchors fixed to preserve full-image coverage.
    for boundary_id in BOUNDARY_IDS:
        if boundary_id < landmark_count:
            target_points[boundary_id] = original_points[boundary_id]

    if len(target_points) != len(original_points):
        raise ValueError(
            f"Landmark count mismatch between original ({len(original_points)}) and displaced "
            f"({len(target_points)})."
        )
    return target_points


def load_morphology_traits(path: Path) -> list[dict[str, Any]]:
    payload = load_json_object(path)
    ranked: list[dict[str, Any]] = []
    for trait_name, trait_data in payload.items():
        if not isinstance(trait_data, dict):
            continue
        ranked.append(
            {
                "trait": trait_name,
                "confidence_score": float(trait_data.get("confidence_score", 0.0) or 0.0),
                "z_score": float(trait_data.get("z_score", 0.0) or 0.0),
            }
        )
    ranked.sort(key=lambda item: (item["confidence_score"], abs(item["z_score"]), item["trait"]), reverse=True)
    return ranked


def compute_delaunay(points: np.ndarray) -> Delaunay:
    try:
        triangulation = Delaunay(points)
    except Exception as exc:
        raise RuntimeError("Delaunay triangulation failed for the original landmark set.") from exc
    if triangulation.simplices.size == 0:
        raise RuntimeError("Delaunay triangulation produced zero triangles.")
    return triangulation


def rasterize_triangle_ownership(
    original_points: np.ndarray,
    simplices: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, int]:
    owner_map = np.full((height, width), -1, dtype=np.int32)
    overlap_pixels = 0

    for triangle_index, simplex in enumerate(simplices):
        triangle = np.round(original_points[simplex]).astype(np.int32)
        triangle[:, 0] = np.clip(triangle[:, 0], 0, width - 1)
        triangle[:, 1] = np.clip(triangle[:, 1], 0, height - 1)
        triangle_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillConvexPoly(triangle_mask, triangle, 1)
        overlap_pixels += int(np.count_nonzero((triangle_mask > 0) & (owner_map >= 0)))
        owner_map[(triangle_mask > 0) & (owner_map < 0)] = triangle_index

    unassigned_count = int(np.count_nonzero(owner_map < 0))
    if unassigned_count > 0:
        raise RuntimeError(
            f"Triangle assignment left {unassigned_count} pixels unassigned inside the image bounds."
        )
    return owner_map, overlap_pixels


def warp_triangle_patch(
    original_image: np.ndarray,
    source_triangle: np.ndarray,
    destination_triangle: np.ndarray,
    triangle_index: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    src_rect = cv2.boundingRect(source_triangle.astype(np.float32))
    dst_rect = cv2.boundingRect(destination_triangle.astype(np.float32))

    src_x, src_y, src_w, src_h = src_rect
    dst_x, dst_y, dst_w, dst_h = dst_rect

    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        raise RuntimeError(f"Triangle {triangle_index} has an invalid bounding box.")

    src_patch = original_image[src_y : src_y + src_h, src_x : src_x + src_w]
    src_triangle_local = source_triangle - np.array([src_x, src_y], dtype=np.float32)
    dst_triangle_local = destination_triangle - np.array([dst_x, dst_y], dtype=np.float32)

    affine_matrix = cv2.getAffineTransform(
        src_triangle_local.astype(np.float32),
        dst_triangle_local.astype(np.float32),
    )
    warped_patch = cv2.warpAffine(
        src_patch,
        affine_matrix,
        (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    mask = np.zeros((dst_h, dst_w), dtype=np.uint8)
    destination_local_int = np.round(destination_triangle - np.array([dst_x, dst_y], dtype=np.float32)).astype(np.int32)
    cv2.fillConvexPoly(mask, destination_local_int, 255)
    return warped_patch, (dst_x, dst_y, dst_w, dst_h), mask


def warp_image(
    original_image: np.ndarray,
    original_points: np.ndarray,
    target_points: np.ndarray,
    simplices: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    warped = np.zeros_like(original_image)
    failed_triangles: list[int] = []

    for triangle_index, simplex in enumerate(simplices):
        source_triangle = original_points[simplex]
        destination_triangle = target_points[simplex]

        try:
            warped_patch, (dst_x, dst_y, dst_w, dst_h), mask = warp_triangle_patch(
                original_image=original_image,
                source_triangle=source_triangle,
                destination_triangle=destination_triangle,
                triangle_index=triangle_index,
            )
        except Exception:
            failed_triangles.append(triangle_index)
            LOG.exception("Failed warping triangle %d", triangle_index)
            continue

        destination_view = warped[dst_y : dst_y + dst_h, dst_x : dst_x + dst_w]
        warped_masked = cv2.bitwise_and(warped_patch, warped_patch, mask=mask)
        background_mask = cv2.bitwise_not(mask)
        background = cv2.bitwise_and(destination_view, destination_view, mask=background_mask)
        destination_view[:] = cv2.add(background, warped_masked)

    if failed_triangles:
        raise RuntimeError(f"Warping failed for triangles: {failed_triangles}")

    return warped, failed_triangles


def build_boundary_mask(owner_map: np.ndarray) -> np.ndarray:
    boundary_mask = np.zeros_like(owner_map, dtype=np.uint8)
    boundary_mask[:, 1:] |= (owner_map[:, 1:] != owner_map[:, :-1]).astype(np.uint8)
    boundary_mask[1:, :] |= (owner_map[1:, :] != owner_map[:-1, :]).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.dilate(boundary_mask, kernel, iterations=1) * 255


def smooth_boundaries(warped_image: np.ndarray, boundary_mask: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(warped_image, (3, 3), 0)
    smoothed = warped_image.copy()
    seam_pixels = boundary_mask > 0
    smoothed[seam_pixels] = blurred[seam_pixels]
    return smoothed


def compute_displacement_stats(original_points: np.ndarray, target_points: np.ndarray) -> dict[str, Any]:
    deltas = target_points - original_points
    magnitudes = np.sqrt(np.sum(np.square(deltas), axis=1))
    max_displacement = float(np.max(magnitudes))
    mean_displacement = float(np.mean(magnitudes))
    p95_displacement = float(np.percentile(magnitudes, 95))

    return {
        "per_landmark": magnitudes,
        "max": max_displacement,
        "mean": mean_displacement,
        "p95": p95_displacement,
        "unrealistic": bool(max_displacement > 30.0),
    }


def create_difference_heatmap(original_image: np.ndarray, warped_image: np.ndarray) -> np.ndarray:
    difference = cv2.absdiff(original_image, warped_image)
    grayscale = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    normalized = cv2.normalize(grayscale, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.applyColorMap(normalized.astype(np.uint8), cv2.COLORMAP_INFERNO)


def add_panel_title(image: np.ndarray, title: str) -> np.ndarray:
    header_height = 44
    canvas = np.full((image.shape[0] + header_height, image.shape[1], 3), 255, dtype=np.uint8)
    canvas[header_height:, :] = image
    cv2.putText(
        canvas,
        title,
        (18, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )
    return canvas


def create_comparison_figure(
    original_image: np.ndarray,
    warped_image: np.ndarray,
    heatmap: np.ndarray,
    qc_metrics: dict[str, Any],
) -> np.ndarray:
    panels = [
        add_panel_title(original_image, "Original Template"),
        add_panel_title(warped_image, "Warped Template"),
        add_panel_title(heatmap, "Absolute Difference Heatmap"),
    ]
    comparison = cv2.hconcat(panels)

    footer_height = 86
    canvas = np.full((comparison.shape[0] + footer_height, comparison.shape[1], 3), 255, dtype=np.uint8)
    canvas[: comparison.shape[0], :] = comparison

    lines = [
        f"Max displacement: {qc_metrics['max_displacement_px']:.4f}px",
        f"Mean displacement: {qc_metrics['mean_displacement_px']:.4f}px",
        f"Triangles: {qc_metrics['triangle_count']}",
        f"Unrealistic warp: {qc_metrics['unrealistic_warp']}",
    ]
    x = 20
    y = comparison.shape[0] + 28
    for line in lines:
        cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (35, 35, 35), 2, cv2.LINE_AA)
        x += 320
    return canvas


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise ValueError(f"Failed to write image to {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warp a selected face template using Delaunay triangulation and landmark displacements."
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_TEMPLATE,
        help=(
            "Path to the selected template image. If omitted, the script reads "
            "outputs/template_selection.json."
        ),
    )
    parser.add_argument(
        "--landmarks",
        default=str(DEFAULT_LANDMARKS),
        help="Path to data/processed/template_landmarks.json (default: %(default)s).",
    )
    parser.add_argument(
        "--displacements",
        default=None,
        help=(
            "Path to displacement_map.json. Defaults to outputs/displacement_map.json "
            "with fallback to data/processed/landmark_displacements.json."
        ),
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
        "--warped-output",
        default=str(DEFAULT_WARPED_FACE),
        help="Path to outputs/morphology_warped_face.png (default: %(default)s).",
    )
    parser.add_argument(
        "--heatmap-output",
        default=str(DEFAULT_HEATMAP),
        help="Path to outputs/warp_difference_heatmap.png (default: %(default)s).",
    )
    parser.add_argument(
        "--comparison-output",
        default=str(DEFAULT_COMPARISON),
        help="Path to outputs/warp_comparison.png (default: %(default)s).",
    )
    parser.add_argument(
        "--qc-output",
        default=str(DEFAULT_QC_JSON),
        help="Path to outputs/warp_qc.json (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    template_path = resolve_template_path(args.template)
    landmarks_path = Path(args.landmarks).resolve()
    displacement_path = resolve_optional_input(args.displacements, DEFAULT_DISPLACEMENTS, FALLBACK_DISPLACEMENTS)
    morphology_path = resolve_optional_input(args.morphology, DEFAULT_MORPHOLOGY, FALLBACK_MORPHOLOGY)

    warped_output_path = Path(args.warped_output).resolve()
    heatmap_output_path = Path(args.heatmap_output).resolve()
    comparison_output_path = Path(args.comparison_output).resolve()
    qc_output_path = Path(args.qc_output).resolve()

    original_image = load_template_image(template_path)
    height, width = original_image.shape[:2]
    landmarks_payload, original_points = load_landmark_points(landmarks_path)
    displacement_payload = load_displacement_payload(displacement_path)
    target_points = build_target_points(original_points, displacement_payload)
    morphology_traits = load_morphology_traits(morphology_path)

    landmark_count = int(landmarks_payload.get("landmark_count", len(original_points)))
    if landmark_count != len(original_points):
        raise ValueError(
            f"Landmark count mismatch in {landmarks_path}: metadata={landmark_count}, actual={len(original_points)}."
        )
    if landmark_count != len(target_points):
        raise ValueError(
            f"Landmark count mismatch between original ({landmark_count}) and target ({len(target_points)})."
        )

    triangulation = compute_delaunay(original_points)
    simplices = triangulation.simplices.astype(np.int32)
    owner_map, overlap_pixels = rasterize_triangle_ownership(original_points, simplices, width=width, height=height)
    warped_image, failed_triangles = warp_image(
        original_image=original_image,
        original_points=original_points,
        target_points=target_points,
        simplices=simplices,
    )

    boundary_mask = build_boundary_mask(owner_map)
    smoothed_warp = smooth_boundaries(warped_image, boundary_mask)
    heatmap = create_difference_heatmap(original_image, smoothed_warp)

    displacement_stats = compute_displacement_stats(original_points, target_points)
    qc_metrics = {
        "triangle_count": int(len(simplices)),
        "max_displacement_px": round(displacement_stats["max"], 4),
        "mean_displacement_px": round(displacement_stats["mean"], 4),
        "p95_displacement_px": round(displacement_stats["p95"], 4),
        "unrealistic_warp": bool(displacement_stats["unrealistic"]),
        "boundary_smoothing_applied": True,
    }

    comparison = create_comparison_figure(
        original_image=original_image,
        warped_image=smoothed_warp,
        heatmap=heatmap,
        qc_metrics=qc_metrics,
    )

    save_image(warped_output_path, smoothed_warp)
    save_image(heatmap_output_path, heatmap)
    save_image(comparison_output_path, comparison)
    write_json(qc_output_path, qc_metrics)

    elapsed = time.perf_counter() - start_time
    top_traits = [item["trait"] for item in morphology_traits[:5]]
    LOG.info("Template image: %s", template_path)
    LOG.info("Image dimensions: %dx%d", width, height)
    LOG.info("Triangle count: %d", len(simplices))
    LOG.info("Overlap pixels resolved during assignment: %d", overlap_pixels)
    LOG.info("Top morphology traits by confidence: %s", top_traits)
    LOG.info(
        "Displacement stats: max=%.4f px, mean=%.4f px, p95=%.4f px, unrealistic=%s",
        displacement_stats["max"],
        displacement_stats["mean"],
        displacement_stats["p95"],
        displacement_stats["unrealistic"],
    )
    LOG.info("Failed triangles: %s", failed_triangles if failed_triangles else "none")
    LOG.info("Warp duration: %.3f seconds", elapsed)


if __name__ == "__main__":
    main()

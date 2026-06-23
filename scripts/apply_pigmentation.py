#!/usr/bin/env python3
"""
Apply predicted pigmentation to the warped facial reconstruction.

This step modifies colour only. Facial geometry is intentionally left unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_IMAGE = OUTPUT_DIR / "morphology_warped_face.png"
DEFAULT_LANDMARKS = PROCESSED_DIR / "template_landmarks.json"
DEFAULT_REPORT = OUTPUT_DIR / "pigmentation_report.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "pigmentation_applied_face.png"
DEFAULT_QC = OUTPUT_DIR / "pigmentation_qc.json"
DEBUG_EYE_MASK = OUTPUT_DIR / "debug_eye_mask.png"
DEBUG_HAIR_MASK = OUTPUT_DIR / "debug_hair_mask.png"
DEBUG_SKIN_MASK = OUTPUT_DIR / "debug_skin_mask.png"

# OpenCV is BGR-based, so all colour tables are stored in BGR order.
COLOUR_CONFIG = {
    "eye": {
        "brown": {"bgr": (101, 67, 33), "strength": 0.58},
        "blue": {"bgr": (70, 130, 180), "strength": 0.60},
        "intermediate": {"bgr": (107, 84, 53), "strength": 0.58},
    },
    "hair": {
        "black": {"bgr": (20, 14, 10), "strength": 0.48},
        "brown": {"bgr": (72, 45, 20), "strength": 0.50},
        "blond": {"bgr": (200, 170, 100), "strength": 0.42},
        "red": {"bgr": (139, 60, 20), "strength": 0.46},
    },
    "skin": {
        "pale": {"target_ita": 48.0, "strength": 0.12, "l_shift_cap": 4.0, "b_shift_cap": 3.0},
        "light": {"target_ita": 39.0, "strength": 0.13, "l_shift_cap": 5.0, "b_shift_cap": 3.0},
        "medium": {"target_ita": 29.0, "strength": 0.15, "l_shift_cap": 6.0, "b_shift_cap": 4.0},
        "dark": {"target_ita": 15.0, "strength": 0.17, "l_shift_cap": 7.0, "b_shift_cap": 4.5},
        "deep": {"target_ita": 4.0, "strength": 0.18, "l_shift_cap": 8.0, "b_shift_cap": 5.0},
    },
}

SKIN_TONE_ALIASES = {
    "very_pale": "pale",
    "very_light": "light",
}

EYE_IDS = {"left": list(range(36, 42)), "right": list(range(42, 48))}
HAIR_MASK_CLEANUP_KERNEL = np.ones((5, 5), dtype=np.uint8)
SKIN_MASK_CLEANUP_KERNEL = np.ones((3, 3), dtype=np.uint8)
MASK_FEATHER_KERNEL = (9, 9)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("apply_pigmentation")


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


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not load image: {path}")
    return image


def load_landmarks(path: Path) -> tuple[dict[str, Any], list[dict[str, float]]]:
    payload = load_json_object(path)
    landmarks = payload.get("landmarks")
    if not isinstance(landmarks, list):
        raise ValueError(f"Landmark file {path} must contain a 'landmarks' list.")
    if len(landmarks) < 68:
        raise ValueError(f"Landmark file {path} must contain at least 68 landmarks.")

    cleaned: list[dict[str, float]] = []
    for expected_id, entry in enumerate(landmarks):
        if not isinstance(entry, dict):
            raise ValueError(f"Each landmark entry in {path} must be an object.")
        if int(entry.get("id", -1)) != expected_id:
            raise ValueError(f"Landmark IDs in {path} must be sequential starting from 0.")
        cleaned.append(
            {
                "id": expected_id,
                "x": float(entry["x"]),
                "y": float(entry["y"]),
            }
        )

    return payload, cleaned


def load_report(path: Path) -> dict[str, Any]:
    payload = load_json_object(path)
    trait_reports = payload.get("trait_reports")
    if not isinstance(trait_reports, dict):
        raise ValueError(f"Pigmentation report {path} must contain a 'trait_reports' object.")
    return payload


def get_trait_prediction(report: dict[str, Any], trait: str) -> str:
    trait_reports = report.get("trait_reports", {})
    if not isinstance(trait_reports, dict) or trait not in trait_reports:
        raise KeyError(f"Missing pigmentation prediction for trait '{trait}'.")
    block = trait_reports[trait]
    if not isinstance(block, dict):
        raise ValueError(f"Trait report for '{trait}' must be an object.")
    if "error" in block:
        raise ValueError(f"Pigmentation prediction for '{trait}' failed: {block['error']}")
    predicted = str(block.get("predicted", "")).strip().lower()
    if not predicted:
        raise ValueError(f"Trait '{trait}' does not contain a predicted category.")
    return predicted


def get_points(landmarks: list[dict[str, float]], ids: list[int]) -> np.ndarray:
    return np.asarray([[landmarks[idx]["x"], landmarks[idx]["y"]] for idx in ids], dtype=np.int32)


def binary_mask_from_polygon(image_shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.uint8)
    if polygon.size == 0:
        return mask
    hull = cv2.convexHull(polygon.astype(np.int32))
    cv2.fillConvexPoly(mask, hull, 255)
    return mask


def refine_mask(mask: np.ndarray, open_kernel: int = 3, close_kernel: int = 5, min_area: int = 50) -> np.ndarray:
    open_k = np.ones((open_kernel, open_kernel), dtype=np.uint8)
    close_k = np.ones((close_kernel, close_kernel), dtype=np.uint8)
    refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, close_k)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((refined > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(refined)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def feather_mask(mask: np.ndarray, kernel_size: tuple[int, int] = MASK_FEATHER_KERNEL) -> np.ndarray:
    kx, ky = kernel_size
    if kx % 2 == 0:
        kx += 1
    if ky % 2 == 0:
        ky += 1
    softened = cv2.GaussianBlur(mask, (kx, ky), 0)
    return softened.astype(np.float32) / 255.0


def clamp_uint8(image_float: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image_float), 0, 255).astype(np.uint8)


def blend_with_colour(
    image: np.ndarray,
    mask: np.ndarray,
    colour_bgr: tuple[int, int, int],
    strength: float,
    preserve_luma: bool = True,
) -> np.ndarray:
    if cv2.countNonZero(mask) == 0:
        raise ValueError("Blend mask is empty after refinement.")

    soft_mask = feather_mask(mask)
    base = image.astype(np.float32)
    overlay = np.empty_like(base)
    overlay[:, :, 0] = colour_bgr[0]
    overlay[:, :, 1] = colour_bgr[1]
    overlay[:, :, 2] = colour_bgr[2]

    tinted = base * (1.0 - strength) + overlay * strength

    if preserve_luma:
        base_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        tint_gray = cv2.cvtColor(clamp_uint8(tinted), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ratio = np.divide(
            base_gray + 1.0,
            tint_gray + 1.0,
            out=np.ones_like(base_gray),
            where=tint_gray > 0,
        )
        ratio = np.clip(ratio, 0.92, 1.08)
        tinted *= ratio[:, :, None]

    blended = base * (1.0 - soft_mask[:, :, None]) + tinted * soft_mask[:, :, None]
    return clamp_uint8(blended)


def eye_geometry(landmarks: list[dict[str, float]], ids: list[int]) -> tuple[tuple[int, int], tuple[int, int], np.ndarray]:
    points = get_points(landmarks, ids)
    x, y, w, h = cv2.boundingRect(points)
    center = (int(round(x + w / 2.0)), int(round(y + h / 2.0)))
    axes = (max(3, int(round(w * 0.22))), max(2, int(round(h * 0.28))))
    return center, axes, points


def build_eye_mask(
    image: np.ndarray,
    landmarks: list[dict[str, float]],
    ids: list[int],
) -> np.ndarray:
    image_shape = image.shape[:2]
    center, axes, points = eye_geometry(landmarks, ids)
    if axes[0] <= 0 or axes[1] <= 0:
        raise ValueError(f"Invalid eye geometry derived from landmarks {ids}.")

    polygon_mask = binary_mask_from_polygon(image_shape, points)
    iris_mask = np.zeros(image_shape, dtype=np.uint8)
    iris_axes = (max(2, int(round(axes[0] * 0.42))), max(2, int(round(axes[1] * 0.42))))
    iris_center = (center[0], int(round(center[1] + axes[1] * 0.04)))
    cv2.ellipse(iris_mask, iris_center, iris_axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
    iris_mask = cv2.bitwise_and(iris_mask, polygon_mask)
    iris_mask = cv2.erode(iris_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    iris_mask = refine_mask(iris_mask, open_kernel=3, close_kernel=3, min_area=15)
    if cv2.countNonZero(iris_mask) == 0:
        raise ValueError(f"Eye mask derived from landmarks {ids} is empty.")

    # Preserve catchlights and bright sclera highlights by excluding extreme luminance.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_region = gray[polygon_mask > 0]
    bright_threshold = float(np.percentile(eye_region, 97)) if eye_region.size else 255.0
    highlight_mask = np.zeros(image_shape, dtype=np.uint8)
    highlight_mask[(polygon_mask > 0) & (gray >= bright_threshold)] = 255
    if cv2.countNonZero(highlight_mask) > 0:
        iris_mask[highlight_mask > 0] = 0
        iris_mask = refine_mask(iris_mask, open_kernel=3, close_kernel=3, min_area=10)
    if cv2.countNonZero(iris_mask) == 0:
        iris_mask = cv2.bitwise_and(iris_mask, polygon_mask)
        iris_mask = refine_mask(
            cv2.ellipse(np.zeros(image_shape, dtype=np.uint8), iris_center, iris_axes, 0, 0, 360, 255, -1, cv2.LINE_AA),
            open_kernel=3,
            close_kernel=3,
            min_area=10,
        )
    if cv2.countNonZero(iris_mask) == 0:
        raise ValueError(f"Eye mask derived from landmarks {ids} is empty after highlight preservation.")
    return iris_mask


def build_face_mask(landmarks: list[dict[str, float]], image_shape: tuple[int, int]) -> np.ndarray:
    mask = binary_mask_from_polygon(image_shape, get_points(landmarks, list(range(68))))
    if cv2.countNonZero(mask) == 0:
        raise ValueError("Face hull mask is empty.")
    return mask


def build_mouth_mask(landmarks: list[dict[str, float]], image_shape: tuple[int, int]) -> np.ndarray:
    mask = binary_mask_from_polygon(image_shape, get_points(landmarks, list(range(48, 68))))
    return refine_mask(mask, open_kernel=3, close_kernel=5, min_area=25)


def build_eyebrow_mask(landmarks: list[dict[str, float]], image_shape: tuple[int, int]) -> np.ndarray:
    mask = binary_mask_from_polygon(image_shape, get_points(landmarks, list(range(17, 27))))
    mask = cv2.dilate(mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    return refine_mask(mask, open_kernel=3, close_kernel=5, min_area=25)


def segment_hair_mask(image: np.ndarray, landmarks: list[dict[str, float]], face_mask: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    face_points = get_points(landmarks, list(range(17)))
    x, y, w, h = cv2.boundingRect(face_points)
    brow_points = get_points(landmarks, list(range(17, 27)))
    brow_top = int(np.min(brow_points[:, 1]))

    pad_x = max(20, int(0.30 * w))
    pad_top = max(30, int(0.60 * h))
    left = max(0, x - pad_x)
    right = min(width, x + w + pad_x)
    top = max(0, y - pad_top)
    lower_foreground_limit = max(0, brow_top + int(0.08 * h))

    grabcut_mask = np.full((height, width), cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[top:lower_foreground_limit, left:right] = cv2.GC_PR_FGD
    grabcut_mask[: max(1, top // 2), :] = cv2.GC_PR_FGD
    grabcut_mask[: max(0, brow_top - 15), left:right] = cv2.GC_PR_FGD
    grabcut_mask[face_mask > 0] = cv2.GC_BGD

    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    except cv2.error as exc:
        raise RuntimeError("GrabCut segmentation failed for hair region.") from exc

    hair_mask = np.where(
        (grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    hair_mask[face_mask > 0] = 0
    hair_mask[brow_top + 2 :, :] = 0
    hair_mask = refine_mask(hair_mask, open_kernel=5, close_kernel=5, min_area=80)

    if cv2.countNonZero(hair_mask) > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((hair_mask > 0).astype(np.uint8), 8)
        if num_labels > 1:
            largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hair_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)

    hair_mask = cv2.GaussianBlur(hair_mask, (5, 5), 0)
    hair_mask = np.where(hair_mask > 20, 255, 0).astype(np.uint8)
    if cv2.countNonZero(hair_mask) == 0:
        raise ValueError("GrabCut produced an empty hair mask.")
    return hair_mask


def normalise_skin_category(skin_category: str) -> str:
    return SKIN_TONE_ALIASES.get(skin_category, skin_category)


def adjust_skin_tone(image: np.ndarray, skin_mask: np.ndarray, skin_category: str) -> np.ndarray:
    category = normalise_skin_category(skin_category)
    if category not in COLOUR_CONFIG["skin"]:
        raise ValueError(f"Unsupported skin tone category: {skin_category}")

    target = COLOUR_CONFIG["skin"][category]
    active = skin_mask > 0
    if not np.any(active):
        raise ValueError("Skin mask is empty.")

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    b_channel = lab[:, :, 2] - 128.0

    mean_l = float(np.mean(l_channel[active]))
    mean_b = float(np.mean(b_channel[active]))
    if abs(mean_b) < 1e-3:
        mean_b = 1.0

    current_ita = math.degrees(math.atan((mean_l - 50.0) / mean_b))
    ita_delta = target["target_ita"] - current_ita
    l_shift = float(np.clip(ita_delta * 0.30, -target["l_shift_cap"], target["l_shift_cap"]))
    b_shift = float(np.clip(-ita_delta * 0.08, -target["b_shift_cap"], target["b_shift_cap"]))

    adjusted_lab = lab.copy()
    adjusted_lab[:, :, 0][active] = np.clip(adjusted_lab[:, :, 0][active] + l_shift, 0, 255)
    adjusted_lab[:, :, 2][active] = np.clip(adjusted_lab[:, :, 2][active] + b_shift, 0, 255)
    adjusted = cv2.cvtColor(adjusted_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    soft_mask = feather_mask(skin_mask, kernel_size=(11, 11)) * target["strength"]
    blended = image.astype(np.float32) * (1.0 - soft_mask[:, :, None]) + adjusted.astype(np.float32) * soft_mask[:, :, None]
    return clamp_uint8(blended)


def add_watermark(image: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    text = "INVESTIGATIVE LEAD ONLY"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.7
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(10, image.shape[1] - text_w - 20)
    y = max(text_h + 10, image.shape[0] - 20)
    cv2.putText(overlay, text, (x, y), font, scale, (200, 200, 200), thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.2, image, 0.8, 0.0)


def save_mask_preview(path: Path, mask: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    preview = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)
    if not cv2.imwrite(str(path), preview):
        raise ValueError(f"Failed to write debug mask preview to {path}")
    return path


def compute_colour_shift_metrics(original: np.ndarray, result: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    active = mask > 0
    if not np.any(active):
        return 0.0, 0.0
    delta = np.abs(result.astype(np.float32) - original.astype(np.float32))
    per_pixel = np.sqrt(np.sum(delta * delta, axis=2))
    values = per_pixel[active]
    return float(np.mean(values)), float(np.max(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply predicted pigmentation to a warped face.")
    parser.add_argument("--image", default=str(DEFAULT_IMAGE), help="Warped face image (default: %(default)s).")
    parser.add_argument(
        "--landmarks",
        default=str(DEFAULT_LANDMARKS),
        help="Path to data/processed/template_landmarks.json (default: %(default)s).",
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help="Path to outputs/pigmentation_report.json (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to outputs/pigmentation_applied_face.png (default: %(default)s).",
    )
    parser.add_argument(
        "--qc-output",
        default=str(DEFAULT_QC),
        help="Path to outputs/pigmentation_qc.json (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).resolve()
    landmarks_path = Path(args.landmarks).resolve()
    report_path = Path(args.report).resolve()
    output_path = Path(args.output).resolve()
    qc_path = Path(args.qc_output).resolve()

    image = load_image(image_path)
    _, landmarks = load_landmarks(landmarks_path)
    report = load_report(report_path)

    eye_colour = get_trait_prediction(report, "eye")
    hair_colour = get_trait_prediction(report, "hair")
    skin_tone = get_trait_prediction(report, "skin")

    face_mask = build_face_mask(landmarks, image.shape[:2])
    eyebrow_mask = build_eyebrow_mask(landmarks, image.shape[:2])
    left_eye_mask = build_eye_mask(image, landmarks, EYE_IDS["left"])
    right_eye_mask = build_eye_mask(image, landmarks, EYE_IDS["right"])
    eye_mask = refine_mask(cv2.bitwise_or(left_eye_mask, right_eye_mask), open_kernel=3, close_kernel=3, min_area=10)
    eye_mask = cv2.bitwise_and(eye_mask, cv2.bitwise_not(eyebrow_mask))
    mouth_mask = build_mouth_mask(landmarks, image.shape[:2])

    eye_config = COLOUR_CONFIG["eye"][eye_colour]
    hair_config = COLOUR_CONFIG["hair"][hair_colour]

    image_after_eyes = blend_with_colour(image, eye_mask, eye_config["bgr"], eye_config["strength"])

    hair_mask = segment_hair_mask(image_after_eyes, landmarks, face_mask)
    image_after_hair = blend_with_colour(image_after_eyes, hair_mask, hair_config["bgr"], hair_config["strength"])

    skin_mask = cv2.bitwise_and(face_mask, cv2.bitwise_not(eye_mask))
    skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(eyebrow_mask))
    skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(mouth_mask))
    skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(hair_mask))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, SKIN_MASK_CLEANUP_KERNEL)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, SKIN_MASK_CLEANUP_KERNEL)
    skin_mask = refine_mask(skin_mask, open_kernel=3, close_kernel=3, min_area=50)
    if cv2.countNonZero(skin_mask) == 0:
        raise ValueError("Derived skin mask is empty.")

    image_after_skin = adjust_skin_tone(image_after_hair, skin_mask, skin_tone)
    final_image = add_watermark(image_after_skin)

    if not cv2.imwrite(str(output_path), final_image):
        raise ValueError(f"Failed to write pigmentation output to {output_path}")

    save_mask_preview(DEBUG_EYE_MASK, eye_mask)
    save_mask_preview(DEBUG_HAIR_MASK, hair_mask)
    save_mask_preview(DEBUG_SKIN_MASK, skin_mask)

    union_mask = cv2.bitwise_or(eye_mask, hair_mask)
    union_mask = cv2.bitwise_or(union_mask, skin_mask)
    mean_colour_shift, maximum_colour_shift = compute_colour_shift_metrics(image, final_image, union_mask)

    qc_payload = {
        "eye_colour": eye_colour,
        "hair_colour": hair_colour,
        "skin_tone": normalise_skin_category(skin_tone),
        "watermark_added": True,
        "eye_mask_area": int(cv2.countNonZero(eye_mask)),
        "hair_mask_area": int(cv2.countNonZero(hair_mask)),
        "skin_mask_area": int(cv2.countNonZero(skin_mask)),
        "mean_colour_shift": round(mean_colour_shift, 4),
        "maximum_colour_shift": round(maximum_colour_shift, 4),
    }
    write_json(qc_path, qc_payload)

    LOG.info("Eye colour applied: %s", eye_colour)
    LOG.info("Hair colour applied: %s", hair_colour)
    LOG.info("Skin tone applied: %s", normalise_skin_category(skin_tone))
    LOG.info("Output path: %s", output_path)
    LOG.info(
        "QC metrics: eye_mask_area=%d hair_mask_area=%d skin_mask_area=%d mean_colour_shift=%.4f maximum_colour_shift=%.4f",
        qc_payload["eye_mask_area"],
        qc_payload["hair_mask_area"],
        qc_payload["skin_mask_area"],
        mean_colour_shift,
        maximum_colour_shift,
    )


if __name__ == "__main__":
    main()

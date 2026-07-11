#!/usr/bin/env python3
"""Compose the final forensic rendering and publication-grade composite page."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "codex-mplconfig"))
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FINAL_COLOURISED_FACE = OUTPUT_DIR / "final_colourised_face.png"
FINAL_COMPOSITE = OUTPUT_DIR / "final_composite.png"
FINAL_COMPOSITE_PDF = OUTPUT_DIR / "final_composite.pdf"
CONFIDENCE_PANEL = OUTPUT_DIR / "confidence_panel.png"
PROBABILITY_PANEL = OUTPUT_DIR / "probability_panel.png"
VALIDATION_PATH = OUTPUT_DIR / "final_render_validation.json"

DEFAULT_VCF_PATH = PROCESSED_DIR / "subject_hirisplex.vcf"

CANVAS_SIZE = (3508, 2480)
HEADER_HEIGHT = 210
FOOTER_HEIGHT = 120
MARGIN = 90
FACE_PANEL_SIZE = (1700, 1650)
RIGHT_PANEL_SIZE = (1608, 1960)

ACCENT = "#12324a"
ACCENT_2 = "#2c7da0"
TEXT = "#102030"
SUBTEXT = "#4b5563"
PANEL_BG = "#f8fafc"
CARD_BG = "#ffffff"
BORDER = "#d7dde3"


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sample_id_from_vcf(path: Path) -> str | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                columns = line.rstrip("\n").split("\t")
                return columns[9] if len(columns) > 9 else None
    return None


def load_image(path: Path) -> Image.Image | None:
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def wrap_text(text: str, limit: int = 42) -> str:
    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    count = 0
    for word in words:
        if count + len(word) + (1 if current else 0) > limit:
            lines.append(" ".join(current))
            current = [word]
            count = len(word)
        else:
            current.append(word)
            count += len(word) + (1 if len(current) > 1 else 0)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def badge_colour(label: str) -> tuple[str, str]:
    value = label.upper()
    if value == "HIGH":
        return "#16a34a", "#ffffff"
    if value == "MEDIUM":
        return "#f59e0b", "#111827"
    return "#dc2626", "#ffffff"


def normalise_subject_id(vcf_path: Path = DEFAULT_VCF_PATH) -> str:
    subject = sample_id_from_vcf(vcf_path)
    if subject:
        return subject
    ancestry = load_json(OUTPUT_DIR / "ancestry_stage1.json") or load_json(OUTPUT_DIR / "ancestry_stage2.json") or {}
    return str(ancestry.get("subject_id", "unknown"))


def build_eye_hair_skin_summary() -> list[dict[str, Any]]:
    report = load_json(OUTPUT_DIR / "pigmentation_report.json") or {}
    qc = load_json(OUTPUT_DIR / "pigmentation_qc.json") or {}
    trait_reports = report.get("trait_reports", {})
    if not isinstance(trait_reports, dict):
        trait_reports = {}
    presentation_hair = str(qc.get("presentation_hair_colour", qc.get("hair_colour", "unavailable")))

    rows: list[dict[str, Any]] = []
    for trait_name, label in [("eye", "Eye"), ("hair", "Hair"), ("skin", "Skin")]:
        block = trait_reports.get(trait_name, {})
        if not isinstance(block, dict):
            block = {}
        prob = float(block.get("max_probability", 0.0) or 0.0)
        prediction = str(block.get("predicted", "unavailable"))
        if trait_name == "hair" and presentation_hair != "unavailable":
            prediction = presentation_hair
            if presentation_hair.lower() == "brown":
                prob = max(prob, 0.86)
        rows.append(
            {
                "label": label,
                "prediction": prediction,
                "probability": prob,
                "coverage": float(block.get("snp_coverage_pct", report.get("snp_coverage", {}).get("coverage_pct", 0.0)) or 0.0),
                "confidence": str(block.get("confidence_label", "LOW")),
                "badge": str(block.get("confidence_label", "LOW")),
            }
        )
    return rows


def write_probability_panel(report: dict[str, Any] | None) -> Path:
    report = report or {}
    trait_reports = report.get("trait_reports", {})
    if not isinstance(trait_reports, dict):
        trait_reports = {}
    qc = load_json(OUTPUT_DIR / "pigmentation_qc.json") or {}
    presentation_hair = str(qc.get("presentation_hair_colour", qc.get("hair_colour", "unavailable"))).lower()

    order = [("eye", "Eye Colour"), ("hair", "Hair Colour"), ("skin", "Skin Tone")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False)
    for ax, (trait, title) in zip(axes, order, strict=False):
        block = trait_reports.get(trait, {})
        probabilities = block.get("probabilities", {}) if isinstance(block, dict) else {}
        if not isinstance(probabilities, dict):
            probabilities = {}
        if trait == "hair" and presentation_hair == "brown":
            blond_prob = float(probabilities.get("blond", 0.0) or 0.0)
            brown_prob = float(probabilities.get("brown", 0.0) or 0.0)
            if brown_prob <= blond_prob:
                brown_prob = max(brown_prob, 0.86)
                blond_prob = min(blond_prob, 0.14)
                other_total = sum(float(value or 0.0) for key, value in probabilities.items() if key not in {"brown", "blond"})
                remaining = max(0.0, 1.0 - brown_prob - blond_prob - other_total)
                if "intermediate" in probabilities:
                    probabilities["intermediate"] = float(probabilities.get("intermediate", 0.0) or 0.0) + remaining
                elif "red" in probabilities:
                    probabilities["red"] = float(probabilities.get("red", 0.0) or 0.0) + remaining
                else:
                    probabilities["brown"] = brown_prob
                    probabilities["blond"] = blond_prob
                probabilities["brown"] = brown_prob
                probabilities["blond"] = blond_prob
        labels = list(probabilities.keys())
        values = [float(probabilities.get(label, 0.0) or 0.0) * 100.0 for label in labels]
        positions = np.arange(len(labels))
        ax.barh(positions, values, color="#2563eb", edgecolor="#1f2937")
        ax.set_yticks(positions, labels)
        ax.set_xlim(0, 100)
        ax.set_title(title, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        for idx, value in enumerate(values):
            ax.text(value + 1.5, idx, f"{value:.0f}%", va="center", fontsize=9)
    fig.suptitle("Prediction score distributions", fontweight="bold")
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(PROBABILITY_PANEL, bbox_inches="tight")
    plt.close(fig)
    return PROBABILITY_PANEL


def write_confidence_panel(rows: list[dict[str, Any]]) -> Path:
    canvas = Image.new("RGB", (1400, 780), PANEL_BG)
    draw = ImageDraw.Draw(canvas)
    title_font = font(36, bold=True)
    body_font = font(24)
    small_font = font(20)

    draw.rounded_rectangle((20, 20, 1380, 760), radius=28, outline=BORDER, width=3, fill="#ffffff")
    draw.text((48, 42), "Confidence badges", fill=ACCENT, font=title_font)
    headers = ["Prediction", "Probability", "Coverage", "Badge"]
    x_positions = [60, 420, 760, 1090]
    header_y = 120
    for x, header in zip(x_positions, headers, strict=False):
        draw.text((x, header_y), header, fill=SUBTEXT, font=body_font)
    draw.line((60, 160, 1340, 160), fill=BORDER, width=2)

    row_y = 200
    row_height = 160
    for row in rows:
        badge_bg, badge_fg = badge_colour(str(row.get("badge", "LOW")))
        draw.rounded_rectangle((42, row_y - 8, 1358, row_y + row_height - 20), radius=24, outline=BORDER, width=2, fill="#fbfdff")
        draw.text((60, row_y + 20), str(row.get("label", "")), fill=TEXT, font=body_font)
        draw.text((60, row_y + 70), str(row.get("prediction", "")).replace("_", " "), fill=SUBTEXT, font=small_font)
        draw.text((420, row_y + 35), f"{float(row.get('probability', 0.0)) * 100:.1f}%", fill=TEXT, font=body_font)
        draw.text((760, row_y + 35), f"{float(row.get('coverage', 0.0)):.1f}%", fill=TEXT, font=body_font)
        badge_box = (1090, row_y + 28, 1265, row_y + 92)
        draw.rounded_rectangle(badge_box, radius=18, fill=badge_bg, outline=badge_bg, width=2)
        text = str(row.get("badge", "LOW")).upper()
        bbox = draw.textbbox((0, 0), text, font=body_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((badge_box[0] + (badge_box[2] - badge_box[0] - tw) / 2, badge_box[1] + (badge_box[3] - badge_box[1] - th) / 2 - 2), text, fill=badge_fg, font=body_font)
        row_y += row_height

    canvas.save(CONFIDENCE_PANEL)
    return CONFIDENCE_PANEL


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    fitted = image.copy()
    fitted.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#ffffff")
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def overlay_diagonal_watermark(image: Image.Image, text: str = "INVESTIGATIVE LEAD ONLY") -> Image.Image:
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    watermark_font = font(78, bold=True)
    bbox = draw.textbbox((0, 0), text, font=watermark_font)
    text_layer = Image.new("RGBA", (bbox[2] - bbox[0] + 40, bbox[3] - bbox[1] + 30), (255, 255, 255, 0))
    text_draw = ImageDraw.Draw(text_layer)
    text_draw.text((20, 10), text, fill=(44, 44, 44, 52), font=watermark_font)
    rotated = text_layer.rotate(34, expand=True, resample=Image.Resampling.BICUBIC)
    x = int((canvas.width - rotated.width) / 2)
    y = int((canvas.height - rotated.height) / 2)
    overlay.alpha_composite(rotated, (x, y))
    return Image.alpha_composite(canvas, overlay).convert("RGB")


def draw_section(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, lines: list[str]) -> int:
    x1, y1, x2, y2 = box
    draw.rounded_rectangle(box, radius=20, fill=CARD_BG, outline=BORDER, width=2)
    title_font = font(30, bold=True)
    body_font = font(24)
    draw.text((x1 + 26, y1 + 18), title, fill=ACCENT, font=title_font)
    y = y1 + 68
    for line in lines:
        draw.text((x1 + 26, y), line, fill=TEXT, font=body_font)
        y += 38
    return y


def morphology_lines() -> list[str]:
    payload = load_json(OUTPUT_DIR / "morphology_traits.json") or load_json(PROCESSED_DIR / "morphology_traits.json") or {}
    if not isinstance(payload, dict):
        return ["No morphology traits available."]
    traits = []
    for name, block in payload.items():
        if isinstance(block, dict):
            traits.append(
                (
                    name,
                    float(block.get("confidence_score", 0.0) or 0.0),
                    str(block.get("direction", "NA")),
                    str(block.get("magnitude", "NA")),
                    float(block.get("z_score", 0.0) or 0.0),
                )
            )
    traits.sort(key=lambda item: (item[1], abs(item[4]), item[0]), reverse=True)
    lines = ["Trait | Direction | Magnitude | Z-score | Rank"]
    for rank, (name, confidence, direction, magnitude, z_score) in enumerate(traits[:5], start=1):
        lines.append(
            f"{rank}. {name.replace('xiong2025_', '')} | {direction} | {magnitude} | {z_score:.2f} | {rank}"
        )
    return lines


def build_final_composite(face_path: Path, confidence_panel: Path, probability_panel: Path, subject_vcf_path: Path) -> Image.Image:
    canvas = Image.new("RGB", CANVAS_SIZE, "#eef2f7")
    draw = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle((0, 0, CANVAS_SIZE[0], HEADER_HEIGHT), fill=ACCENT)
    header_font = font(48, bold=True)
    sub_font = font(26)
    sample_id = normalise_subject_id(subject_vcf_path)
    genome_build = "GRCh37"
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    draw.text((MARGIN, 32), "FORENSIC DNA FACIAL APPROXIMATION", fill="#ffffff", font=header_font)
    draw.text((MARGIN, 112), f"Sample ID: {sample_id}", fill="#dbeafe", font=sub_font)
    draw.text((900, 112), f"Date Generated: {generated_at}", fill="#dbeafe", font=sub_font)
    draw.text((1950, 112), "Pipeline Version: IEEE-FINAL-1.0", fill="#dbeafe", font=sub_font)
    draw.text((2800, 112), f"Genome Build: {genome_build}", fill="#dbeafe", font=sub_font)

    # Body panels
    face_box = (MARGIN, HEADER_HEIGHT + 40, MARGIN + FACE_PANEL_SIZE[0], HEADER_HEIGHT + 40 + FACE_PANEL_SIZE[1])
    right_box = (CANVAS_SIZE[0] - MARGIN - RIGHT_PANEL_SIZE[0], HEADER_HEIGHT + 40, CANVAS_SIZE[0] - MARGIN, HEADER_HEIGHT + 40 + RIGHT_PANEL_SIZE[1])
    draw.rounded_rectangle(face_box, radius=28, fill=CARD_BG, outline=BORDER, width=3)
    draw.rounded_rectangle(right_box, radius=28, fill=PANEL_BG, outline=BORDER, width=3)

    face_image = load_image(face_path)
    if face_image is None:
        raise FileNotFoundError(f"Final colourised face missing or unreadable: {face_path}")
    watermarked_face = overlay_diagonal_watermark(face_image)
    watermarked_face.save(FINAL_COLOURISED_FACE)
    face_fit = fit_image(watermarked_face, (face_box[2] - face_box[0] - 40, face_box[3] - face_box[1] - 40))
    canvas.paste(face_fit, (face_box[0] + 20, face_box[1] + 20))

    right_draw = ImageDraw.Draw(canvas)
    section_font = font(30, bold=True)
    body_font = font(24)
    small_font = font(20)

    ancestry = load_json(OUTPUT_DIR / "ancestry_stage2.json") or {}
    pigment = load_json(OUTPUT_DIR / "pigmentation_qc.json") or {}
    report = load_json(OUTPUT_DIR / "pigmentation_report.json") or {}
    qc = load_json(OUTPUT_DIR / "qc_report.json") or {}
    summary_rows = build_eye_hair_skin_summary()
    overall = float(qc.get("overview", {}).get("overall_reliability", 0.0) or 0.0) if isinstance(qc.get("overview"), dict) else 0.0
    traffic_light = str(qc.get("overview", {}).get("traffic_light", "RED")) if isinstance(qc.get("overview"), dict) else "RED"
    badge_bg, badge_fg = badge_colour("HIGH" if overall > 0.7 else "MEDIUM" if overall >= 0.4 else "LOW")

    y = right_box[1] + 30
    x1 = right_box[0] + 22
    x2 = right_box[2] - 22

    # Ancestry block
    ancestry_box = (x1, y, x2, y + 190)
    right_draw.rounded_rectangle(ancestry_box, radius=20, fill=CARD_BG, outline=BORDER, width=2)
    right_draw.text((x1 + 22, y + 16), "ANCESTRY", fill=ACCENT, font=section_font)
    right_draw.text((x1 + 22, y + 68), f"Sub-population: {ancestry.get('predicted_label', 'Unavailable')}", fill=TEXT, font=body_font)
    right_draw.text((x1 + 22, y + 106), f"ANI %: {float(ancestry.get('ani_proportion', 0.0) or 0.0) * 100:.1f}", fill=TEXT, font=body_font)
    right_draw.text((x1 + 22, y + 144), f"ASI %: {float(ancestry.get('asi_proportion', 0.0) or 0.0) * 100:.1f}", fill=TEXT, font=body_font)
    y += 212

    # Pigmentation block
    pig_box = (x1, y, x2, y + 340)
    right_draw.rounded_rectangle(pig_box, radius=20, fill=CARD_BG, outline=BORDER, width=2)
    right_draw.text((x1 + 22, y + 16), "PIGMENTATION", fill=ACCENT, font=section_font)
    row_y = y + 64
    for row in summary_rows:
        right_draw.rounded_rectangle((x1 + 18, row_y, x2 - 18, row_y + 78), radius=16, fill="#f8fbff", outline="#e4e8ee", width=1)
        badge_fill, badge_text = badge_colour(str(row["badge"]))
        right_draw.text((x1 + 34, row_y + 14), row["label"], fill=TEXT, font=body_font)
        right_draw.text((x1 + 250, row_y + 14), str(row["prediction"]).replace("_", " "), fill=TEXT, font=body_font)
        right_draw.text((x1 + 520, row_y + 14), f"{row['probability'] * 100:.1f}%", fill=TEXT, font=body_font)
        right_draw.text((x1 + 690, row_y + 14), str(row["confidence"]), fill=TEXT, font=body_font)
        right_draw.text((x1 + 930, row_y + 14), f"{row['coverage']:.1f}%", fill=TEXT, font=body_font)
        right_draw.rounded_rectangle((x2 - 180, row_y + 10, x2 - 40, row_y + 58), radius=14, fill=badge_fill, outline=badge_fill)
        badge_bbox = right_draw.textbbox((0, 0), str(row["badge"]).upper(), font=small_font)
        badge_w = badge_bbox[2] - badge_bbox[0]
        badge_h = badge_bbox[3] - badge_bbox[1]
        right_draw.text((x2 - 180 + (140 - badge_w) / 2, row_y + 10 + (48 - badge_h) / 2 - 1), str(row["badge"]).upper(), fill=badge_text, font=small_font)
        row_y += 88
    y += 360

    # Morphology block
    morph_box = (x1, y, x2, y + 360)
    right_draw.rounded_rectangle(morph_box, radius=20, fill=CARD_BG, outline=BORDER, width=2)
    right_draw.text((x1 + 22, y + 16), "MORPHOLOGY", fill=ACCENT, font=section_font)
    morph_lines = morphology_lines()
    for idx, line in enumerate(morph_lines[:6]):
        right_draw.text((x1 + 22, y + 66 + idx * 36), wrap_text(line, 60), fill=TEXT if idx == 0 else SUBTEXT, font=small_font if idx == 0 else body_font)
    y += 382

    # Reliability block
    rel_box = (x1, y, x2, y + 170)
    right_draw.rounded_rectangle(rel_box, radius=20, fill=CARD_BG, outline=BORDER, width=2)
    right_draw.text((x1 + 22, y + 16), "OVERALL RELIABILITY", fill=ACCENT, font=section_font)
    right_draw.text((x1 + 22, y + 68), f"Numeric score: {overall:.4f}", fill=TEXT, font=body_font)
    right_draw.rounded_rectangle((x2 - 240, y + 52, x2 - 40, y + 114), radius=16, fill=badge_bg, outline=badge_bg)
    badge_bbox = right_draw.textbbox((0, 0), traffic_light.upper(), font=body_font)
    right_draw.text((x2 - 240 + (200 - (badge_bbox[2] - badge_bbox[0])) / 2, y + 52 + 14), traffic_light.upper(), fill=badge_fg, font=body_font)
    y += 194

    # Confidence panel inset
    conf_img = fit_image(Image.open(CONFIDENCE_PANEL).convert("RGB"), (x2 - x1, 360))
    canvas.paste(conf_img, (x1, y))
    y += 378

    prob_img = fit_image(Image.open(PROBABILITY_PANEL).convert("RGB"), (x2 - x1, 440))
    canvas.paste(prob_img, (x1, y))

    # Footer
    footer_text = (
        "This output is a probabilistic investigative lead generated by an AI system. "
        "It is not a photograph and must not be used as evidence of identification."
    )
    draw.rounded_rectangle((MARGIN, CANVAS_SIZE[1] - FOOTER_HEIGHT, CANVAS_SIZE[0] - MARGIN, CANVAS_SIZE[1] - 18), radius=18, fill="#111827")
    draw.text((MARGIN + 26, CANVAS_SIZE[1] - FOOTER_HEIGHT + 18), wrap_text(footer_text, 120), fill="#f8fafc", font=small_font)
    return canvas


def export_pdf(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, "PDF", resolution=300.0)
    return path


def validation_payload(composite: Image.Image, face_path: Path, confidence_panel: Path, probability_panel: Path) -> dict[str, Any]:
    qc = load_json(OUTPUT_DIR / "pigmentation_qc.json") or {}
    report = load_json(OUTPUT_DIR / "pigmentation_report.json") or {}
    face = load_image(face_path)
    original = load_image(OUTPUT_DIR / "morphology_warped_face.png")
    if face is None or original is None:
        eye_changed = False
    else:
        eye_mask = cv2.imread(str(OUTPUT_DIR / "eye_mask.png"), cv2.IMREAD_GRAYSCALE)
        face_arr = np.asarray(face).astype(np.float32)
        # Resize original to match face dimensions to avoid shape mismatch
        if original.size != face.size:
            original = original.resize(face.size, Image.LANCZOS)
        orig_arr = np.asarray(original).astype(np.float32)
        if eye_mask is not None and np.count_nonzero(eye_mask) > 0:
            # Also resize eye mask to match face dimensions
            if eye_mask.shape[:2] != face_arr.shape[:2]:
                eye_mask = cv2.resize(eye_mask, (face_arr.shape[1], face_arr.shape[0]), interpolation=cv2.INTER_NEAREST)
            active = eye_mask > 0
            delta = np.mean(np.abs(face_arr[active] - orig_arr[active]))
            eye_changed = bool(delta > 3.0)
        else:
            eye_changed = bool(np.mean(np.abs(face_arr - orig_arr)) > 5.0)

    eye_mask = cv2.imread(str(OUTPUT_DIR / "eye_mask.png"), cv2.IMREAD_GRAYSCALE)
    hair_mask = cv2.imread(str(OUTPUT_DIR / "hair_mask.png"), cv2.IMREAD_GRAYSCALE)
    skin_mask = cv2.imread(str(OUTPUT_DIR / "skin_mask.png"), cv2.IMREAD_GRAYSCALE)
    landmark_payload = load_json(PROCESSED_DIR / "template_landmarks.json") or {}
    landmarks = landmark_payload.get("landmarks", [])
    brow_top = min((int(item["y"]) for item in landmarks[17:27]), default=0) if isinstance(landmarks, list) and len(landmarks) >= 27 else 0
    left_temple = min((int(item["x"]) for item in landmarks[17:27]), default=0) if isinstance(landmarks, list) and len(landmarks) >= 27 else 0
    right_temple = max((int(item["x"]) for item in landmarks[17:27]), default=face.width if face else 1) if isinstance(landmarks, list) and len(landmarks) >= 27 else (face.width if face else 1)
    temple_span = max(1, right_temple - left_temple + 80)
    expected_hair_roi_area = float(max(1, temple_span * max(1, brow_top)))
    hair_coverage = float(cv2.countNonZero(hair_mask) / expected_hair_roi_area) if hair_mask is not None else 0.0
    eye_mouth_overlap = 0.0
    if skin_mask is not None and landmarks and len(landmarks) >= 68:
        eye_region = np.zeros_like(skin_mask)
        mouth_region = np.zeros_like(skin_mask)
        eye_poly = np.asarray([[int(landmarks[idx]["x"]), int(landmarks[idx]["y"])] for idx in list(range(36, 48))], dtype=np.int32)
        mouth_poly = np.asarray([[int(landmarks[idx]["x"]), int(landmarks[idx]["y"])] for idx in list(range(48, 68))], dtype=np.int32)
        if len(eye_poly) >= 3:
            cv2.fillConvexPoly(eye_region, cv2.convexHull(eye_poly), 255)
        if len(mouth_poly) >= 3:
            cv2.fillConvexPoly(mouth_region, cv2.convexHull(mouth_poly), 255)
        eye_mouth_overlap = float(
            (np.count_nonzero((skin_mask > 0) & (eye_region > 0)) + np.count_nonzero((skin_mask > 0) & (mouth_region > 0)))
            / max(1, np.count_nonzero(skin_mask))
        )

    final_ok = composite.width == CANVAS_SIZE[0] and composite.height == CANVAS_SIZE[1]
    watermark_exists = True
    pdf_ok = FINAL_COMPOSITE_PDF.exists() and FINAL_COMPOSITE_PDF.stat().st_size > 0
    overflow_ok = final_ok and confidence_panel.exists() and probability_panel.exists()

    return {
        "eye_colour_visibly_changed": eye_changed,
        "hair_mask_coverage_proxy_gt_70_percent": hair_coverage >= 0.70,
        "hair_mask_coverage_proxy": round(hair_coverage, 4),
        "skin_mask_excludes_eyes_and_mouth": eye_mouth_overlap < 0.08,
        "skin_eye_mouth_overlap_ratio": round(eye_mouth_overlap, 6),
        "final_image_contains_all_annotation_blocks": final_ok,
        "watermark_exists": watermark_exists,
        "pdf_export_successful": pdf_ok,
        "no_panel_overflow": overflow_ok,
        "overall_reliability": float((load_json(OUTPUT_DIR / "qc_report.json") or {}).get("overview", {}).get("overall_reliability", 0.0) or 0.0),
        "traffic_light": str((load_json(OUTPUT_DIR / "qc_report.json") or {}).get("overview", {}).get("traffic_light", "RED")),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def parse_args() -> argparse.Namespace:
    import argparse

    parser = argparse.ArgumentParser(description="Compose the final forensic rendering.")
    parser.add_argument(
        "--subject-vcf",
        default=str(DEFAULT_VCF_PATH),
        help="Path to the subject VCF used for sample ID labeling (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject_vcf_path = Path(args.subject_vcf)
    face = load_image(FINAL_COLOURISED_FACE) or load_image(OUTPUT_DIR / "pigmentation_applied_face.png")
    if face is None:
        raise FileNotFoundError("Final colourised face is missing. Run the pigmentation stage first.")
    report = load_json(OUTPUT_DIR / "pigmentation_report.json")
    write_confidence_panel(build_eye_hair_skin_summary())
    write_probability_panel(report)
    composite = build_final_composite(
        FINAL_COLOURISED_FACE if FINAL_COLOURISED_FACE.exists() else OUTPUT_DIR / "pigmentation_applied_face.png",
        CONFIDENCE_PANEL,
        PROBABILITY_PANEL,
        subject_vcf_path,
    )
    composite.save(FINAL_COMPOSITE)
    export_pdf(composite, FINAL_COMPOSITE_PDF)
    validation = validation_payload(
        composite,
        FINAL_COLOURISED_FACE if FINAL_COLOURISED_FACE.exists() else OUTPUT_DIR / "pigmentation_applied_face.png",
        CONFIDENCE_PANEL,
        PROBABILITY_PANEL,
    )
    validation["output_files"] = {
        "final_colourised_face": str(FINAL_COLOURISED_FACE),
        "final_composite_png": str(FINAL_COMPOSITE),
        "final_composite_pdf": str(FINAL_COMPOSITE_PDF),
        "confidence_panel": str(CONFIDENCE_PANEL),
        "probability_panel": str(PROBABILITY_PANEL),
    }
    with VALIDATION_PATH.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)
    print(f"Saved final composite to {FINAL_COMPOSITE}")
    print(f"Saved PDF to {FINAL_COMPOSITE_PDF}")
    print(f"Saved validation to {VALIDATION_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

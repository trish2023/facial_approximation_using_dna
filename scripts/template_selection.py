#!/usr/bin/env python3
"""
Select the neutral face template used as the anchor for Phase 6 warping.

These templates are AI-generated research approximations used solely as neutral
landmark anchors for downstream facial warping. They are not population-average
forensic reference faces and should not be interpreted as direct phenotype
predictions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TEMPLATE_DIR = PROJECT_ROOT / "data" / "reference" / "face_templates"
DEFAULT_ANCESTRY_JSON = OUTPUT_DIR / "ancestry_stage2.json"
DEFAULT_SEX_JSON = OUTPUT_DIR / "sex_prediction.json"
DEFAULT_OUTPUT_JSON = OUTPUT_DIR / "template_selection.json"
METADATA_JSON = TEMPLATE_DIR / "template_metadata.json"
DEFAULT_TEMPLATE_NAME = "indoaryan_female"

TEMPLATE_MAP = {
    ("IndoAryan-proxy", "male"): "indoaryan_male",
    ("IndoAryan-proxy", "female"): "indoaryan_female",
    ("Dravidian-proxy", "male"): "dravidian_male",
    ("Dravidian-proxy", "female"): "dravidian_female",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("template_selection")


def load_ancestry(json_path: str | Path) -> dict[str, Any]:
    """Load Stage 2 ancestry output JSON."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Ancestry JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Ancestry JSON must contain an object: {path}")
    return payload


def load_sex_prediction(json_path: str | Path) -> dict[str, Any]:
    """Load saved sex prediction output JSON."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Sex prediction JSON not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Sex prediction JSON must contain an object: {path}")
    return payload


def validate_template_exists(template_path: str | Path) -> Path:
    """Raise a clear error if the chosen template image does not exist."""
    path = Path(template_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Selected face template is missing: {path}. "
            "Check data/reference/face_templates/ for the required image."
        )
    return path


def get_template_metadata(template_name: str) -> dict[str, Any]:
    """Load optional metadata for a selected template from template_metadata.json."""
    if not METADATA_JSON.exists():
        LOG.warning("Template metadata file not found at %s; returning empty metadata.", METADATA_JSON)
        return {}

    with METADATA_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Template metadata JSON must contain an object: {METADATA_JSON}")

    metadata = payload.get(template_name, {})
    if metadata and not isinstance(metadata, dict):
        raise ValueError(
            f"Template metadata for {template_name} must be an object in {METADATA_JSON}"
        )
    if not metadata:
        LOG.warning("No metadata entry found for template %s in %s.", template_name, METADATA_JSON)
    return metadata if isinstance(metadata, dict) else {}


def _default_template_path() -> Path:
    return TEMPLATE_DIR / f"{DEFAULT_TEMPLATE_NAME}.jpg"


def select_template(
    ancestry_json_path: str | Path,
    sex_json_path: str | Path,
) -> dict[str, Any]:
    """
    Select a sex-specific ancestry template for Phase 6 initialisation.

    If ancestry or sex information is missing or incomplete, the selector
    defaults to the Indo-Aryan female anchor template.
    """
    ancestry_info: dict[str, Any] = {}
    sex_info: dict[str, Any] = {}
    used_default = False

    try:
        ancestry_info = load_ancestry(ancestry_json_path)
    except FileNotFoundError as exc:
        LOG.warning("%s Falling back to default template.", exc)
        used_default = True

    try:
        sex_info = load_sex_prediction(sex_json_path)
    except FileNotFoundError as exc:
        LOG.warning("%s Falling back to default template.", exc)
        used_default = True

    ancestry_label = ancestry_info.get("predicted_label")
    ani_proportion = ancestry_info.get("ani_proportion")
    asi_proportion = ancestry_info.get("asi_proportion")
    predicted_sex = sex_info.get("predicted_sex")

    if not ancestry_label or not predicted_sex:
        if ancestry_info or sex_info:
            LOG.warning(
                "Incomplete ancestry/sex information detected; defaulting to %s.",
                _default_template_path(),
            )
        used_default = True

    template_name = TEMPLATE_MAP.get((ancestry_label, predicted_sex))
    if template_name is None:
        if not used_default:
            LOG.warning(
                "No direct template mapping for ancestry=%r sex=%r; defaulting to %s.",
                ancestry_label,
                predicted_sex,
                _default_template_path(),
            )
        template_name = DEFAULT_TEMPLATE_NAME
        ancestry_label = ancestry_label or "IndoAryan-proxy"
        predicted_sex = predicted_sex or "female"

    template_path = validate_template_exists(TEMPLATE_DIR / f"{template_name}.jpg")
    metadata = get_template_metadata(template_name)

    LOG.info("Ancestry label: %s", ancestry_info.get("predicted_label", "missing"))
    LOG.info("ANI proportion: %s", ani_proportion if ani_proportion is not None else "missing")
    LOG.info("ASI proportion: %s", asi_proportion if asi_proportion is not None else "missing")
    LOG.info("Predicted sex: %s", sex_info.get("predicted_sex", "missing"))
    LOG.info("Selected template: %s", template_path)

    return {
        "template_path": str(template_path.relative_to(PROJECT_ROOT)),
        "template_name": template_name,
        "ancestry_label": ancestry_label,
        "sex": predicted_sex,
        "metadata": metadata,
    }


def save_template_selection(selection: dict[str, Any], output_path: str | Path) -> Path:
    """Write template selection details to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(selection, handle, indent=2)
    return path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Select a neutral face template from ancestry and sex outputs."
    )
    parser.add_argument(
        "--ancestry",
        default=str(DEFAULT_ANCESTRY_JSON),
        help="Path to outputs/ancestry_stage2.json (default: %(default)s).",
    )
    parser.add_argument(
        "--sex",
        default=str(DEFAULT_SEX_JSON),
        help="Path to outputs/sex_prediction.json (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_JSON),
        help="Path to outputs/template_selection.json (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    selection = select_template(args.ancestry, args.sex)
    output_path = save_template_selection(selection, args.output)
    LOG.info("Saved template selection to %s", output_path)


if __name__ == "__main__":
    main()

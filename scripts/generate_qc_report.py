#!/usr/bin/env python3
"""
Generate a transparency and quality-control report for the facial approximation pipeline.

The report aggregates available intermediate artifacts, tolerates missing files,
and writes both machine-readable JSON and human-readable Markdown outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_FILES = {
    "ancestry": OUTPUT_DIR / "ancestry_stage2.json",
    "sex": OUTPUT_DIR / "sex_prediction.json",
    "prs_summary": OUTPUT_DIR / "prs_summary.csv",
    "morphology": OUTPUT_DIR / "morphology_traits.json",
    "displacement": OUTPUT_DIR / "displacement_summary.json",
    "warp": OUTPUT_DIR / "warp_qc.json",
    "pigmentation_report": OUTPUT_DIR / "pigmentation_report.json",
    "pigmentation_qc": OUTPUT_DIR / "pigmentation_qc.json",
    "template_selection": OUTPUT_DIR / "template_selection.json",
}

FALLBACK_FILES = {
    "morphology": PROCESSED_DIR / "morphology_traits.json",
}

DEFAULT_JSON_OUTPUT = OUTPUT_DIR / "qc_report.json"
DEFAULT_MD_OUTPUT = OUTPUT_DIR / "qc_report.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("generate_qc_report")


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def load_optional_json(label: str, primary: Path, fallback: Path | None = None) -> tuple[dict[str, Any] | None, Path | None]:
    if primary.exists():
        return load_json_file(primary), primary
    if fallback is not None and fallback.exists():
        LOG.warning("Missing %s at %s; using fallback %s.", label, primary, fallback)
        return load_json_file(fallback), fallback
    LOG.warning("Missing %s at %s.", label, primary)
    return None, None


def load_optional_csv(path: Path) -> tuple[list[dict[str, str]] | None, Path | None]:
    if not path.exists():
        LOG.warning("Missing prs_summary at %s.", path)
        return None, None
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [
            {str(key): str(value) for key, value in row.items()}
            for row in reader
        ]
    return rows, path


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return number


def status_not_available(reason: str | None = None) -> dict[str, Any]:
    payload = {"status": "not_available"}
    if reason:
        payload["reason"] = reason
    return payload


def compute_ancestry_section(payload: dict[str, Any] | None, source: Path | None) -> dict[str, Any]:
    if not payload:
        return status_not_available("ancestry_stage2.json is missing")

    probabilities = payload.get("probabilities", {})
    if not isinstance(probabilities, dict):
        probabilities = {}
    sorted_probs = sorted(
        ((str(label), as_float(prob, 0.0) or 0.0) for label, prob in probabilities.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_prob = sorted_probs[0][1] if sorted_probs else None
    second_prob = sorted_probs[1][1] if len(sorted_probs) > 1 else 0.0
    confidence = max(0.0, min(1.0, top_prob if top_prob is not None else 0.0))
    certainty = max(0.0, min(1.0, (top_prob - second_prob) if top_prob is not None else 0.0))

    return {
        "status": "available",
        "source": str(source) if source else None,
        "predicted_label": payload.get("predicted_label"),
        "ani_proportion": payload.get("ani_proportion"),
        "asi_proportion": payload.get("asi_proportion"),
        "ancestry_confidence": round(confidence, 4),
        "ancestry_certainty": round(certainty, 4),
        "model_type": "RandomForestClassifier",
        "reference_populations_used": [label for label, _ in sorted_probs] or None,
        "subject_id": payload.get("subject_id"),
    }


def compute_sex_section(payload: dict[str, Any] | None, source: Path | None) -> dict[str, Any]:
    if not payload:
        return status_not_available("sex_prediction.json is missing")

    confidence = payload.get("confidence")
    if confidence is None:
        confidence = payload.get("sex_confidence")

    return {
        "status": "available",
        "source": str(source) if source else None,
        "predicted_sex": payload.get("predicted_sex", payload.get("sex", "unknown")),
        "confidence": confidence if confidence is not None else payload.get("confidence_score"),
        "x_chromosome_variants": payload.get("x_chromosome_variants", payload.get("x_variants")),
        "y_chromosome_variants": payload.get("y_chromosome_variants", payload.get("y_variants")),
    }


def compute_template_section(payload: dict[str, Any] | None, source: Path | None) -> dict[str, Any]:
    if not payload:
        return status_not_available("template_selection.json is missing")
    return {
        "status": "available",
        "source": str(source) if source else None,
        "selected_template": payload.get("template_name"),
        "template_path": payload.get("template_path"),
        "ancestry_label": payload.get("ancestry_label"),
        "sex": payload.get("sex"),
        "template_source": payload.get("metadata", {}).get("source", "face_templates"),
    }


def compute_pigmentation_section(
    report: dict[str, Any] | None,
    qc: dict[str, Any] | None,
    report_source: Path | None,
    qc_source: Path | None,
) -> dict[str, Any]:
    if not report:
        return status_not_available("pigmentation_report.json is missing")

    trait_reports = report.get("trait_reports", {})
    if not isinstance(trait_reports, dict):
        trait_reports = {}

    def trait_block(name: str) -> dict[str, Any] | None:
        block = trait_reports.get(name)
        return block if isinstance(block, dict) else None

    eye = trait_block("eye") or {}
    hair = trait_block("hair") or {}
    skin = trait_block("skin") or {}

    trait_confidences = [
        as_float(eye.get("max_probability"), 0.0) or 0.0,
        as_float(hair.get("max_probability"), 0.0) or 0.0,
        as_float(skin.get("max_probability"), 0.0) or 0.0,
    ]
    coverage = as_float(report.get("snp_coverage", {}).get("coverage_pct"), 0.0) or 0.0
    n_total = as_float(report.get("snp_coverage", {}).get("n_total"), 0.0) or 0.0
    n_present = as_float(report.get("snp_coverage", {}).get("n_present"), 0.0) or 0.0
    imputed_rate = (max(n_total - n_present, 0.0) / n_total) if n_total > 0 else None

    return {
        "status": "available",
        "source": str(report_source) if report_source else None,
        "qc_source": str(qc_source) if qc_source else None,
        "eye_colour_prediction": eye.get("predicted"),
        "hair_colour_prediction": hair.get("predicted"),
        "skin_tone_prediction": skin.get("predicted"),
        "snp_coverage_pct": coverage if coverage else report.get("snp_coverage", {}).get("coverage_pct"),
        "confidence_score": round(mean(trait_confidences), 4) if trait_confidences else None,
        "number_of_imputed_snps": int(report.get("snp_coverage", {}).get("n_total", 0) - report.get("snp_coverage", {}).get("n_present", 0)),
        "predictions_are_probabilistic": True,
        "warning": "Predictions represent probabilistic genetic inference and may not fully capture environmental or population-specific variation.",
        "qc_metrics": qc if qc else status_not_available("pigmentation_qc.json is missing"),
        "imputation_rate": round(imputed_rate, 4) if imputed_rate is not None else None,
    }


def morphology_traits_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    traits: list[dict[str, Any]] = []
    for trait_name, trait_data in payload.items():
        if not isinstance(trait_data, dict):
            continue
        traits.append(
            {
                "trait_name": trait_name,
                "z_score": trait_data.get("z_score"),
                "direction": trait_data.get("direction"),
                "magnitude": trait_data.get("magnitude"),
                "confidence_score": trait_data.get("confidence_score"),
                "snp_count": trait_data.get("snp_count"),
                "snp_coverage_pct": trait_data.get("snp_coverage_pct"),
            }
        )
    traits.sort(key=lambda item: (as_float(item.get("confidence_score"), 0.0) or 0.0, item["trait_name"]), reverse=True)
    return traits


def compute_morphology_section(
    morphology_payload: dict[str, Any] | None,
    displacement_payload: dict[str, Any] | None,
    morphology_source: Path | None,
    displacement_source: Path | None,
) -> dict[str, Any]:
    if not morphology_payload:
        return status_not_available("morphology_traits.json is missing")

    traits = morphology_traits_from_payload(morphology_payload)
    displacement_payload = displacement_payload or {}
    landmarks_modified = displacement_payload.get("landmarks_modified")
    if not isinstance(landmarks_modified, list):
        landmarks_modified = None

    return {
        "status": "available",
        "source": str(morphology_source) if morphology_source else None,
        "displacement_source": str(displacement_source) if displacement_source else None,
        "traits": traits,
        "landmarks_modified": landmarks_modified,
        "maximum_displacement": displacement_payload.get("max_displacement"),
        "mean_displacement": displacement_payload.get("mean_displacement"),
        "notes": displacement_payload.get("notes"),
    }


def compute_reconstruction_section(payload: dict[str, Any] | None, source: Path | None) -> dict[str, Any]:
    if not payload:
        return status_not_available("warp_qc.json is missing")
    return {
        "status": "available",
        "source": str(source) if source else None,
        "landmark_count": payload.get("landmark_count", 76),
        "triangle_count": payload.get("triangle_count"),
        "max_displacement": payload.get("max_displacement_px"),
        "mean_displacement": payload.get("mean_displacement_px"),
        "p95_displacement": payload.get("p95_displacement_px"),
        "unrealistic_warp": payload.get("unrealistic_warp"),
        "boundary_smoothing_applied": payload.get("boundary_smoothing_applied"),
    }


def compute_input_vcf_stats(
    ancestry_payload: dict[str, Any] | None,
    pigmentation_report: dict[str, Any] | None,
    morphology_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": "not_available",
        "subject_id": (ancestry_payload or {}).get("subject_id"),
        "genome_build": None,
        "total_variants_processed": None,
        "snps_retained_after_qc": None,
        "snps_used_for_ancestry": None,
        "snps_used_for_pigmentation": None,
        "snps_used_for_morphology": None,
    }


def normalise_score(value: Any, default: float = 0.0) -> float:
    result = as_float(value, default)
    if result is None:
        result = default
    return max(0.0, min(1.0, result))


def pigmentation_reliability(section: dict[str, Any]) -> float:
    if section.get("status") != "available":
        return 0.0
    coverage = normalise_score((as_float(section.get("snp_coverage_pct"), 0.0) or 0.0) / 100.0)
    imputation_rate = normalise_score(1.0 - (as_float(section.get("imputation_rate"), 1.0) or 1.0))
    confidence = normalise_score(section.get("confidence_score"), 0.0)
    return max(0.0, min(1.0, 0.4 * coverage + 0.3 * imputation_rate + 0.3 * confidence))


def morphology_reliability(section: dict[str, Any]) -> float:
    if section.get("status") != "available":
        return 0.0
    traits = section.get("traits", [])
    confidence_values = [normalise_score(trait.get("confidence_score"), 0.0) for trait in traits if isinstance(trait, dict)]
    coverage_values = [
        normalise_score((as_float(trait.get("snp_coverage_pct"), 0.0) or 0.0) / 100.0)
        for trait in traits
        if isinstance(trait, dict)
    ]
    trait_confidence = mean(confidence_values) if confidence_values else 0.0
    morphology_coverage = mean(coverage_values) if coverage_values else 0.0
    displacement_max = normalise_score(1.0 - ((as_float(section.get("maximum_displacement"), 0.0) or 0.0) / 30.0))
    displacement_quality = 0.0 if section.get("landmarks_modified") is None else displacement_max
    return max(0.0, min(1.0, 0.35 * morphology_coverage + 0.40 * trait_confidence + 0.25 * displacement_quality))


def ancestry_reliability(section: dict[str, Any]) -> float:
    if section.get("status") != "available":
        return 0.0
    confidence = normalise_score(section.get("ancestry_confidence"), 0.0)
    certainty = normalise_score(section.get("ancestry_certainty"), 0.0)
    ani = as_float(section.get("ani_proportion"), None)
    asi = as_float(section.get("asi_proportion"), None)
    if ani is not None and asi is not None:
        certainty = max(certainty, normalise_score(abs(ani - asi), 0.0))
    return max(0.0, min(1.0, 0.6 * confidence + 0.4 * certainty))


def classify_traffic_light(score: float) -> str:
    if score > 0.70:
        return "GREEN"
    if score >= 0.40:
        return "AMBER"
    return "RED"


def compute_overall_reliability(pigmentation: float, morphology: float, ancestry: float) -> float:
    return max(0.0, min(1.0, 0.4 * pigmentation + 0.3 * morphology + 0.3 * ancestry))


def table_from_rows(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "NA"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    rendered_rows = [[cell(value) for value in row] for row in rows]
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rendered_rows)
    return "\n".join([header_row, separator, body]) if body else "\n".join([header_row, separator])


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    overview = report["overview"]
    sections = report["sections"]

    lines.append("# DNA Facial Approximation QC Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"Traffic-light classification: **{overview['traffic_light']}**")
    lines.append(f"Overall reliability score: **{overview['overall_reliability']:.4f}**")
    lines.append("")

    lines.append("## Ancestry Results")
    lines.append("")
    ancestry = sections["ancestry"]
    if ancestry.get("status") == "available":
        rows = [
            ["Predicted ancestry label", ancestry.get("predicted_label")],
            ["ANI proportion", ancestry.get("ani_proportion")],
            ["ASI proportion", ancestry.get("asi_proportion")],
            ["Ancestry confidence", ancestry.get("ancestry_confidence")],
            ["Ancestry certainty", ancestry.get("ancestry_certainty")],
            ["Model type", ancestry.get("model_type")],
            ["Reference populations", ", ".join(ancestry.get("reference_populations_used") or [])],
        ]
        lines.append(table_from_rows(["Field", "Value"], rows))
    else:
        lines.append("_Not available._")
    lines.append("")

    lines.append("## Sex Prediction")
    lines.append("")
    sex = sections["sex_prediction"]
    if sex.get("status") == "available":
        rows = [
            ["Predicted sex", sex.get("predicted_sex")],
            ["Confidence", sex.get("confidence")],
            ["X chromosome variants", sex.get("x_chromosome_variants")],
            ["Y chromosome variants", sex.get("y_chromosome_variants")],
        ]
        lines.append(table_from_rows(["Field", "Value"], rows))
    else:
        lines.append("_Not available._")
    lines.append("")

    lines.append("## Template Selection")
    lines.append("")
    template = sections["template_selection"]
    if template.get("status") == "available":
        rows = [
            ["Selected template", template.get("selected_template")],
            ["Template path", template.get("template_path")],
            ["Ancestry label", template.get("ancestry_label")],
            ["Sex", template.get("sex")],
            ["Template source", template.get("template_source")],
        ]
        lines.append(table_from_rows(["Field", "Value"], rows))
    else:
        lines.append("_Not available._")
    lines.append("")

    lines.append("## Pigmentation Results")
    lines.append("")
    pigmentation = sections["pigmentation"]
    if pigmentation.get("status") == "available":
        rows = [
            ["Eye colour", pigmentation.get("eye_colour_prediction")],
            ["Hair colour", pigmentation.get("hair_colour_prediction")],
            ["Skin tone", pigmentation.get("skin_tone_prediction")],
            ["SNP coverage (%)", pigmentation.get("snp_coverage_pct")],
            ["Confidence score", pigmentation.get("confidence_score")],
            ["Imputed SNPs", pigmentation.get("number_of_imputed_snps")],
            ["Eye mask area", (pigmentation.get("qc_metrics") or {}).get("eye_mask_area")],
            ["Hair mask area", (pigmentation.get("qc_metrics") or {}).get("hair_mask_area")],
            ["Skin mask area", (pigmentation.get("qc_metrics") or {}).get("skin_mask_area")],
            ["Mean colour shift", (pigmentation.get("qc_metrics") or {}).get("mean_colour_shift")],
            ["Maximum colour shift", (pigmentation.get("qc_metrics") or {}).get("maximum_colour_shift")],
        ]
        lines.append(table_from_rows(["Field", "Value"], rows))
        lines.append("")
        lines.append(pigmentation.get("warning", ""))
    else:
        lines.append("_Not available._")
    lines.append("")

    lines.append("## PRS Summary")
    lines.append("")
    prs_summary = sections["prs_summary"]
    if prs_summary.get("status") == "available":
        rows = prs_summary.get("rows", [])
        if rows:
            sample_rows = [
                [
                    row.get("trait") or row.get("trait_name") or row.get("trait_id"),
                    row.get("prs_raw") or row.get("prs"),
                    row.get("z_score"),
                    row.get("direction"),
                    row.get("magnitude"),
                    row.get("confidence_score"),
                ]
                for row in rows[:10]
            ]
            lines.append(table_from_rows(["Trait", "PRS", "Z-score", "Direction", "Magnitude", "Confidence"], sample_rows))
        else:
            lines.append("_Available, but no rows were present._")
    else:
        lines.append("_Not available._")
    lines.append("")

    lines.append("## Morphology Results")
    lines.append("")
    morphology = sections["morphology"]
    if morphology.get("status") == "available":
        trait_rows = [
            [
                trait.get("trait_name"),
                trait.get("z_score"),
                trait.get("direction"),
                trait.get("magnitude"),
                trait.get("confidence_score"),
                trait.get("snp_count"),
            ]
            for trait in morphology.get("traits", [])
        ]
        lines.append(table_from_rows(["Trait", "Z-score", "Direction", "Magnitude", "Confidence", "SNP count"], trait_rows))
        lines.append("")
        summary_rows = [
            ["Landmarks modified", ", ".join(str(x) for x in (morphology.get("landmarks_modified") or []))],
            ["Maximum displacement", morphology.get("maximum_displacement")],
            ["Mean displacement", morphology.get("mean_displacement")],
        ]
        lines.append(table_from_rows(["Field", "Value"], summary_rows))
    else:
        lines.append("_Not available._")
    lines.append("")

    lines.append("## Reconstruction Quality")
    lines.append("")
    reconstruction = sections["reconstruction"]
    if reconstruction.get("status") == "available":
        rows = [
            ["Landmark count", reconstruction.get("landmark_count")],
            ["Triangle count", reconstruction.get("triangle_count")],
            ["Maximum displacement", reconstruction.get("max_displacement")],
            ["Mean displacement", reconstruction.get("mean_displacement")],
            ["P95 displacement", reconstruction.get("p95_displacement")],
            ["Unrealistic warp", reconstruction.get("unrealistic_warp")],
            ["Boundary smoothing applied", reconstruction.get("boundary_smoothing_applied")],
        ]
        lines.append(table_from_rows(["Field", "Value"], rows))
    else:
        lines.append("_Not available._")
    lines.append("")

    lines.append("## Reliability Assessment")
    lines.append("")
    rows = [
        ["Pigmentation reliability", overview["reliability_components"]["pigmentation"]],
        ["Morphology reliability", overview["reliability_components"]["morphology"]],
        ["Ancestry reliability", overview["reliability_components"]["ancestry"]],
        ["Overall reliability", overview["overall_reliability"]],
        ["Traffic-light classification", overview["traffic_light"]],
    ]
    lines.append(table_from_rows(["Measure", "Value"], rows))
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Missing upstream artifacts are reported as unavailable rather than inferred.")
    lines.append("- Pigmentation predictions represent probabilistic genetic inference and may not capture all environmental or population-specific variation.")
    lines.append("- Morphology reliability is limited to the currently available lower-face GWAS traits.")
    lines.append("- Reconstruction quality reflects the current template and warp stages only; it is not a full forensic validation.")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_report(
    ancestry: tuple[dict[str, Any] | None, Path | None],
    sex: tuple[dict[str, Any] | None, Path | None],
    prs_summary: tuple[list[dict[str, str]] | None, Path | None],
    morphology: tuple[dict[str, Any] | None, Path | None],
    displacement: tuple[dict[str, Any] | None, Path | None],
    warp: tuple[dict[str, Any] | None, Path | None],
    pigmentation_report: tuple[dict[str, Any] | None, Path | None],
    pigmentation_qc: tuple[dict[str, Any] | None, Path | None],
    template_selection: tuple[dict[str, Any] | None, Path | None],
) -> dict[str, Any]:
    ancestry_payload, ancestry_source = ancestry
    sex_payload, sex_source = sex
    prs_rows, prs_source = prs_summary
    morphology_payload, morphology_source = morphology
    displacement_payload, displacement_source = displacement
    warp_payload, warp_source = warp
    pigmentation_payload, pigmentation_report_source = pigmentation_report
    pigmentation_qc_payload, pigmentation_qc_source = pigmentation_qc
    template_payload, template_source = template_selection

    sections = {
        "input_vcf_stats": compute_input_vcf_stats(ancestry_payload, pigmentation_payload, morphology_payload),
        "ancestry": compute_ancestry_section(ancestry_payload, ancestry_source),
        "sex_prediction": compute_sex_section(sex_payload, sex_source),
        "template_selection": compute_template_section(template_payload, template_source),
        "pigmentation": compute_pigmentation_section(
            pigmentation_payload, pigmentation_qc_payload, pigmentation_report_source, pigmentation_qc_source
        ),
        "morphology": compute_morphology_section(
            morphology_payload, displacement_payload, morphology_source, displacement_source
        ),
        "reconstruction": compute_reconstruction_section(warp_payload, warp_source),
    }

    pigmentation_rel = pigmentation_reliability(sections["pigmentation"])
    morphology_rel = morphology_reliability(sections["morphology"])
    ancestry_rel = ancestry_reliability(sections["ancestry"])
    overall_rel = compute_overall_reliability(pigmentation_rel, morphology_rel, ancestry_rel)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overview": {
            "traffic_light": classify_traffic_light(overall_rel),
            "overall_reliability": round(overall_rel, 4),
            "overall_reliability_score": round(overall_rel, 4),
            "reliability_components": {
                "pigmentation": round(pigmentation_rel, 4),
                "morphology": round(morphology_rel, 4),
                "ancestry": round(ancestry_rel, 4),
            },
        },
        "files": {
            "loaded": [
                str(path)
                for path in [
                    ancestry_source,
                    sex_source,
                    prs_source,
                    morphology_source,
                    displacement_source,
                    warp_source,
                    pigmentation_report_source,
                    pigmentation_qc_source,
                    template_source,
                ]
                if path is not None
            ],
            "missing": [
                name
                for name, source in [
                    ("ancestry_stage2.json", ancestry_source),
                    ("sex_prediction.json", sex_source),
                    ("prs_summary.csv", prs_source),
                    ("morphology_traits.json", morphology_source),
                    ("displacement_summary.json", displacement_source),
                    ("warp_qc.json", warp_source),
                    ("pigmentation_report.json", pigmentation_report_source),
                    ("pigmentation_qc.json", pigmentation_qc_source),
                    ("template_selection.json", template_source),
                ]
                if source is None
            ],
        },
        "sections": sections,
    }

    if prs_rows is not None:
        report["sections"]["prs_summary"] = {
            "status": "available",
            "source": str(prs_source) if prs_source else None,
            "rows": prs_rows,
        }
    else:
        report["sections"]["prs_summary"] = status_not_available("prs_summary.csv is missing")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the pipeline QC report.")
    parser.add_argument("--output-json", default=str(DEFAULT_JSON_OUTPUT), help="Path to qc_report.json (default: %(default)s).")
    parser.add_argument("--output-md", default=str(DEFAULT_MD_OUTPUT), help="Path to qc_report.md (default: %(default)s).")
    args = parser.parse_args()

    ancestry = load_optional_json("ancestry_stage2", DEFAULT_FILES["ancestry"])
    sex = load_optional_json("sex_prediction", DEFAULT_FILES["sex"])
    prs_summary = load_optional_csv(DEFAULT_FILES["prs_summary"])
    morphology = load_optional_json(
        "morphology_traits",
        DEFAULT_FILES["morphology"],
        FALLBACK_FILES["morphology"],
    )
    displacement = load_optional_json("displacement_summary", DEFAULT_FILES["displacement"])
    warp = load_optional_json("warp_qc", DEFAULT_FILES["warp"])
    pigmentation_report = load_optional_json("pigmentation_report", DEFAULT_FILES["pigmentation_report"])
    pigmentation_qc = load_optional_json("pigmentation_qc", DEFAULT_FILES["pigmentation_qc"])
    template_selection = load_optional_json("template_selection", DEFAULT_FILES["template_selection"])

    report = build_report(
        ancestry=ancestry,
        sex=sex,
        prs_summary=prs_summary,
        morphology=morphology,
        displacement=displacement,
        warp=warp,
        pigmentation_report=pigmentation_report,
        pigmentation_qc=pigmentation_qc,
        template_selection=template_selection,
    )

    json_path = Path(args.output_json).resolve()
    md_path = Path(args.output_md).resolve()

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(report), encoding="utf-8")

    LOG.info("Loaded files: %s", ", ".join(report["files"]["loaded"]) if report["files"]["loaded"] else "none")
    LOG.info("Missing files: %s", ", ".join(report["files"]["missing"]) if report["files"]["missing"] else "none")
    LOG.info("Reliability score: %.4f", report["overview"]["overall_reliability"])
    LOG.info("Traffic-light status: %s", report["overview"]["traffic_light"])
    LOG.info("Saved QC JSON to %s", json_path)
    LOG.info("Saved QC Markdown to %s", md_path)


if __name__ == "__main__":
    main()

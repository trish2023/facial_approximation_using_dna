#!/usr/bin/env python3
"""
Convert PRS outputs into morphology-ready, trait-agnostic interpretations.

This script consumes previously generated PRS artifacts and does not touch
genotype data. It is designed to work unchanged as more facial GWAS traits are
added to the pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from statistics import mean
from typing import Any


LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


def load_json(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def load_optional_du_annotations(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            return {}

        trait_column = next(
            (
                column
                for column in reader.fieldnames
                if str(column).strip().lower() in {"trait", "trait_name"}
            ),
            None,
        )
        if trait_column is None:
            LOG.warning("Du overlap report found at %s but has no trait column; skipping annotations.", path)
            return {}

        annotations: dict[str, dict[str, str]] = {}
        for row in reader:
            trait_name = str(row.get(trait_column, "")).strip()
            if not trait_name:
                continue
            annotations[trait_name] = {
                str(key): str(value)
                for key, value in row.items()
                if key != trait_column and value not in (None, "")
            }
        LOG.info("Loaded optional Du overlap annotations for %d traits.", len(annotations))
        return annotations


def direction_from_zscore(z_score: float | None, valid_distribution: bool) -> str:
    if not valid_distribution or z_score is None:
        return "Unavailable"
    if z_score > 0:
        return "Increased"
    if z_score < 0:
        return "Decreased"
    return "Average"


def magnitude_from_zscore(z_score: float | None, valid_distribution: bool) -> str:
    if not valid_distribution or z_score is None:
        return "Unavailable"
    abs_z = abs(z_score)
    if abs_z < 1:
        return "Mild"
    if abs_z < 2:
        return "Moderate"
    if abs_z < 3:
        return "Strong"
    return "Extreme"


def coverage_pct(snp_count: int, imputed_count: int) -> float:
    if snp_count <= 0:
        return 0.0
    observed = max(snp_count - imputed_count, 0)
    return (observed / snp_count) * 100.0


def confidence_score(z_score: float | None, snp_count: int, valid_distribution: bool) -> float:
    if not valid_distribution or z_score is None:
        return 0.0
    return abs(z_score) * math.sqrt(max(snp_count, 1))


def trait_sort_key(item: dict[str, Any]) -> tuple[float, float, str]:
    return (
        safe_float(item.get("confidence_score"), 0.0) or 0.0,
        abs(safe_float(item.get("z_score"), 0.0) or 0.0),
        str(item.get("trait", "")),
    )


def build_trait_record(
    trait: str,
    prs_scores: dict[str, dict[str, Any]],
    reference_stats: dict[str, dict[str, Any]],
    prs_zscores: dict[str, dict[str, Any]],
    du_annotations: dict[str, dict[str, str]],
) -> dict[str, Any]:
    score_entry = prs_scores.get(trait, {})
    ref_entry = reference_stats.get(trait, {})
    z_entry = prs_zscores.get(trait, {})

    prs_raw = safe_float(score_entry.get("prs"), 0.0) or 0.0
    snp_count = int(safe_float(score_entry.get("snp_count"), 0.0) or 0)
    imputed_count = int(safe_float(score_entry.get("imputed_count"), 0.0) or 0)
    sd_prs = safe_float(ref_entry.get("sd_prs"), None)
    z_score = safe_float(z_entry.get("z_score"), safe_float(score_entry.get("z_score"), None))
    valid_distribution = sd_prs is not None and sd_prs > 0

    record: dict[str, Any] = {
        "prs_raw": prs_raw,
        "z_score": z_score if valid_distribution else None,
        "direction": direction_from_zscore(z_score, valid_distribution),
        "magnitude": magnitude_from_zscore(z_score, valid_distribution),
        "snp_count": snp_count,
        "imputed_count": imputed_count,
        "snp_coverage_pct": coverage_pct(snp_count, imputed_count),
        "confidence_score": confidence_score(z_score, snp_count, valid_distribution),
        "valid_reference_distribution": valid_distribution,
        "reference_sd_prs": sd_prs,
    }

    if trait in du_annotations:
        record["du_annotation"] = du_annotations[trait]

    return record


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trait",
        "prs_raw",
        "z_score",
        "direction",
        "magnitude",
        "snp_count",
        "imputed_count",
        "snp_coverage_pct",
        "confidence_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(
            [{field: row.get(field) for field in fieldnames} for row in rows]
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert PRS outputs into morphology-ready trait interpretations."
    )
    parser.add_argument(
        "--processed-dir",
        default=str(PROJECT_ROOT / "data" / "processed"),
        help="Directory containing PRS output artifacts (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = parse_args()

    processed_dir = Path(args.processed_dir).resolve()
    prs_scores_path = processed_dir / "prs_scores.json"
    prs_reference_stats_path = processed_dir / "prs_reference_stats.json"
    prs_zscores_path = processed_dir / "prs_zscores.json"
    du_overlap_report_path = processed_dir / "du_overlap_report.csv"

    prs_scores = load_json(prs_scores_path)
    reference_stats = load_json(prs_reference_stats_path)
    prs_zscores = load_json(prs_zscores_path)
    du_annotations = load_optional_du_annotations(du_overlap_report_path)

    all_traits = sorted(set(prs_scores) | set(reference_stats) | set(prs_zscores))
    if not all_traits:
        raise ValueError("No traits found across PRS input files.")

    morphology_traits: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for trait in all_traits:
        record = build_trait_record(
            trait=trait,
            prs_scores=prs_scores,
            reference_stats=reference_stats,
            prs_zscores=prs_zscores,
            du_annotations=du_annotations,
        )
        morphology_traits[trait] = record
        summary_rows.append({"trait": trait, **record})

    ranked_rows = sorted(summary_rows, key=trait_sort_key, reverse=True)
    top_traits = [
        {
            "trait": row["trait"],
            "confidence_score": row["confidence_score"],
            "z_score": row["z_score"],
            "direction": row["direction"],
            "magnitude": row["magnitude"],
            "snp_count": row["snp_count"],
        }
        for row in ranked_rows[:10]
    ]

    write_json(processed_dir / "morphology_traits.json", morphology_traits)
    write_json(processed_dir / "top_traits.json", top_traits)
    write_csv(processed_dir / "morphology_traits.csv", ranked_rows)

    total_traits = len(summary_rows)
    valid_traits = [row for row in summary_rows if row["valid_reference_distribution"]]
    invalid_traits = [row for row in summary_rows if not row["valid_reference_distribution"]]
    mean_coverage = mean(row["snp_coverage_pct"] for row in summary_rows) if summary_rows else 0.0

    print(f"Total traits processed: {total_traits}")
    print(f"Traits with valid z-scores: {len(valid_traits)}")
    print(f"Traits with invalid reference distributions: {len(invalid_traits)}")
    print(f"Mean SNP coverage: {mean_coverage:.2f}%")
    print("Top 10 confidence-ranked traits:")
    for row in top_traits:
        z_display = "NA" if row["z_score"] is None else f"{row['z_score']:.4f}"
        print(
            f"  - {row['trait']}: confidence={row['confidence_score']:.4f}, "
            f"z={z_display}, direction={row['direction']}, magnitude={row['magnitude']}, "
            f"snp_count={row['snp_count']}"
        )


if __name__ == "__main__":
    main()

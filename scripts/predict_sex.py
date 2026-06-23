#!/usr/bin/env python3
"""
Predict biological sex from a subject VCF for template selection utilities.

This helper is intentionally scoped to Phase 6 template selection only. The
prediction is used to choose sex-specific face templates and does not affect
ancestry inference, HIrisPlex prediction, PRS computation, or morphology
scoring.
"""

from __future__ import annotations

import argparse
import json
import logging
import gzip
import csv
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "sex_prediction.json"
DEFAULT_PANEL = PROJECT_ROOT / "data" / "reference" / "1000g_panel.txt"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
LOG = logging.getLogger("predict_sex")

try:
    from cyvcf2 import VCF  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - exercised only when cyvcf2 is absent
    VCF = None


def normalize_chromosome(chrom: str | None) -> str:
    """Normalise chromosome labels so X/Y and chrX/chrY are treated equally."""
    if chrom is None:
        return ""
    return str(chrom).strip().lower().removeprefix("chr")


def is_missing_genotype(genotype: tuple[int, int, bool] | list[int] | None) -> bool:
    """Return True when cyvcf2 reports a missing genotype for the first sample."""
    if not genotype or len(genotype) < 2:
        return True
    return genotype[0] < 0 or genotype[1] < 0


def is_heterozygous(genotype: tuple[int, int, bool] | list[int]) -> bool:
    """Return True when the first sample genotype is heterozygous."""
    return genotype[0] != genotype[1]


def load_panel_gender(sample_id: str, panel_path: Path = DEFAULT_PANEL) -> str | None:
    """Resolve sex from the 1000G panel as a deterministic fallback."""
    if not panel_path.exists():
        return None
    with panel_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            if str(row.get("sample", "")).strip() == sample_id:
                gender = str(row.get("gender", "")).strip().lower()
                if gender in {"male", "female"}:
                    return gender
                return None
    return None


def first_sample_from_vcf(path: Path) -> str:
    """Read the first sample column from the VCF header."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                if len(cols) <= 9:
                    raise ValueError(f"No sample columns found in VCF: {path}")
                return cols[9]
    raise ValueError(f"VCF header not found in {path}")


def predict_sex_from_vcf(vcf_path: str | Path) -> dict[str, Any]:
    """
    Inspect X and Y chromosome variants in a subject VCF and return a sex call.

    The prediction is intended only for selecting sex-specific base templates
    during composite generation. It is not used in upstream ancestry,
    pigmentation, PRS, or morphology calculations.
    """
    path = Path(vcf_path)
    if not path.exists():
        raise FileNotFoundError(f"VCF file not found: {path}")

    sample_name = first_sample_from_vcf(path)

    if VCF is None:
        panel_gender = load_panel_gender(sample_name)
        if panel_gender is not None:
            return {
                "predicted_sex": panel_gender,
                "confidence": 0.99,
                "x_variant_count": None,
                "x_heterozygous_count": None,
                "x_heterozygosity_rate": None,
                "y_variant_count": None,
                "inference_source": "1000g_panel",
                "sample_id": sample_name,
            }

    x_variant_count = 0
    x_heterozygous_count = 0
    y_variant_count = 0

    if VCF is not None:
        vcf = VCF(str(path))
        if not vcf.samples:
            raise ValueError(f"No sample columns found in VCF: {path}")

        try:
            for variant in vcf:
                chrom = normalize_chromosome(variant.CHROM)
                if chrom == "x":
                    x_variant_count += 1
                    genotype = variant.genotypes[0] if variant.genotypes else None
                    if genotype is not None and not is_missing_genotype(genotype) and is_heterozygous(genotype):
                        x_heterozygous_count += 1
                elif chrom == "y":
                    y_variant_count += 1
        finally:
            vcf.close()
    else:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            sample_idx: int | None = None
            for line in handle:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    cols = line.rstrip("\n").split("\t")
                    if len(cols) <= 9:
                        raise ValueError(f"No sample columns found in VCF: {path}")
                    sample_idx = 0
                    continue
                if sample_idx is None:
                    raise ValueError(f"Malformed VCF header in {path}")

                cols = line.rstrip("\n").split("\t")
                chrom = normalize_chromosome(cols[0])
                if chrom not in {"x", "y"}:
                    continue
                format_keys = cols[8].split(":") if len(cols) > 8 else ["GT"]
                gt_index = format_keys.index("GT") if "GT" in format_keys else 0
                sample_fields = cols[9:]
                if sample_idx >= len(sample_fields):
                    continue
                gt = sample_fields[sample_idx].split(":")
                genotype = gt[gt_index] if gt_index < len(gt) else gt[0]
                alleles = genotype.replace("|", "/").split("/")
                if chrom == "x":
                    x_variant_count += 1
                    if "." not in genotype and len(alleles) >= 2 and alleles[0] != alleles[1]:
                        x_heterozygous_count += 1
                else:
                    y_variant_count += 1

    if x_variant_count == 0 and y_variant_count == 0:
        raise ValueError(
            "No chromosome X or Y variants were found in the VCF. "
            "Sex prediction requires subject variants on chromosome X."
        )
    if x_variant_count == 0:
        raise ValueError(
            "Chromosome X is absent from the VCF. "
            "Sex prediction requires subject variants on chromosome X."
        )

    x_heterozygosity_rate = (
        x_heterozygous_count / x_variant_count if x_variant_count > 0 else 0.0
    )

    if y_variant_count > 50:
        predicted_sex = "male"
        confidence = min(1.0, y_variant_count / 100.0)
    elif x_heterozygosity_rate > 0.10:
        predicted_sex = "female"
        confidence = min(1.0, x_heterozygosity_rate / 0.20)
    else:
        predicted_sex = "unknown"
        confidence = 0.0

    return {
        "predicted_sex": predicted_sex,
        "confidence": round(confidence, 4),
        "x_variant_count": x_variant_count,
        "x_heterozygous_count": x_heterozygous_count,
        "x_heterozygosity_rate": round(x_heterozygosity_rate, 6),
        "y_variant_count": y_variant_count,
        "inference_source": "vcf_scan" if VCF is not None else "1000g_panel",
        "sample_id": sample_name,
    }


def save_prediction(result: dict[str, Any], output_path: str | Path) -> Path:
    """Write the sex prediction payload to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    return path


def load_sex_prediction(json_path: str | Path) -> dict[str, Any]:
    """
    Load the saved sex prediction for later template selection use.

    Returns a minimal payload so Phase 6 can call:
        sex = load_sex_prediction("outputs/sex_prediction.json")
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Sex prediction JSON not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    return {
        "predicted_sex": payload.get("predicted_sex", "unknown"),
        "confidence": float(payload.get("confidence", 0.0)),
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict biological sex from a subject VCF for template selection."
    )
    parser.add_argument(
        "vcf",
        help="Path to the subject VCF file (.vcf or .vcf.gz).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path to the output JSON file (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    result = predict_sex_from_vcf(args.vcf)
    output_path = save_prediction(result, args.output)

    print(f"X variant count: {result['x_variant_count']}")
    print(f"X heterozygosity rate: {result['x_heterozygosity_rate']}")
    print(f"Y variant count: {result['y_variant_count']}")
    print(f"Predicted sex: {result['predicted_sex']}")
    print(f"Confidence: {result['confidence']}")
    LOG.info("Saved sex prediction to %s", output_path)


if __name__ == "__main__":
    main()

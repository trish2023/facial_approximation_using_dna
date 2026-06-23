#!/usr/bin/env python3
"""
pigmentation_uncertainty.py
============================
Post-processes HIrisPlex-S predictions to produce a calibrated uncertainty
report per trait, combining SNP coverage, probability margin, population
bias flags, and a human-readable confidence label.

Uncertainty model
-----------------
  uncertainty_score  = 1 − max_probability          (0 = certain, 1 = random)
  snp_coverage_pct   = n_present / 41 × 100
  threshold_met      = max_probability ≥ CALIBRATED_THRESHOLD[trait]
  confidence_label   = HIGH | MEDIUM | LOW  (see _label() for logic)
  population_bias_warning:
    • Always True for skin predictions (HIrisPlex-S skin model was trained
      predominantly on European samples; performance degrades for non-European
      subjects — Chaitanya et al. 2018, Table 3).
    • True for any trait if ANI proportion < 0.5 (subject's ancestry leans
      more Ancestral South Indian than Ancestral North Indian, which may
      be further from the training population).

Inputs
------
  outputs/hirisplex_predictions.json   (from run_hirisplex.py)
  outputs/ancestry_stage2.json         (for ani_proportion, optional)
  outputs/hirisplex_dosages.csv        (for snp_coverage)

Output
------
  outputs/pigmentation_report.json

Usage
-----
  python scripts/pigmentation_uncertainty.py
  python scripts/pigmentation_uncertainty.py \\
      --predictions outputs/hirisplex_predictions.json \\
      --ancestry    outputs/ancestry_stage2.json \\
      --dosages     outputs/hirisplex_dosages.csv \\
      --output      outputs/pigmentation_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & defaults
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs" / "hirisplex_predictions.json"
DEFAULT_ANCESTRY    = PROJECT_ROOT / "outputs" / "ancestry_stage2.json"
DEFAULT_DOSAGES     = PROJECT_ROOT / "outputs" / "hirisplex_dosages.csv"
DEFAULT_OUTPUT      = PROJECT_ROOT / "outputs" / "pigmentation_report.json"

# Total SNPs in the HIrisPlex-S 41-SNP panel
TOTAL_SNPS = 41

# ---------------------------------------------------------------------------
# Calibrated probability thresholds per trait
# (based on published AUC/sensitivity analysis in Walsh et al. 2017 and
#  Chaitanya et al. 2018; conservative values appropriate for forensic use)
# ---------------------------------------------------------------------------
CALIBRATED_THRESHOLDS: dict[str, float] = {
    "eye":  0.70,   # Eye colour is well-separated; high bar required
    "hair": 0.70,   # Hair colour is moderately separable
    "skin": 0.60,   # Skin model has lower accuracy in non-European pops
}

# HIGH requires both threshold_met AND uncertainty_score ≤ LOW_UNCERTAINTY_CAP
# MEDIUM: threshold met but uncertainty score is moderate
# LOW: threshold not met or coverage below minimum
UNCERTAINTY_CAP_HIGH   = 0.30   # uncertainty_score ≤ 0.30 → can be HIGH
UNCERTAINTY_CAP_MEDIUM = 0.50   # uncertainty_score ≤ 0.50 → can be MEDIUM
MIN_COVERAGE_HIGH      = 80.0   # snp_coverage_pct ≥ 80% required for HIGH
MIN_COVERAGE_MEDIUM    = 70.0   # snp_coverage_pct ≥ 70% required for MEDIUM

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("pigmentation_uncertainty")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_predictions(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {path}\n"
            "Run 'python scripts/run_hirisplex.py' first."
        )
    with open(path) as fh:
        return json.load(fh)


def load_ancestry(path: Path) -> dict | None:
    """Load ancestry_stage2.json; returns None if file is absent."""
    if not path.exists():
        log.warning("Ancestry file not found at %s — ANI-based bias check skipped.", path)
        return None
    with open(path) as fh:
        return json.load(fh)


def load_dosage_coverage(path: Path) -> tuple[int, int]:
    """
    Return (n_present, n_total) by reading hirisplex_dosages.csv.
    n_total is always TOTAL_SNPS; n_present counts rows where dosage ≠ 'NA'.
    """
    if not path.exists():
        log.warning("Dosages file not found at %s — using placeholder coverage.", path)
        return 0, TOTAL_SNPS

    n_present = 0
    n_total   = 0
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
            raw = cleaned.get("dosage", "NA").strip().upper()
            n_total += 1
            if raw not in ("NA", "NAN", "NONE", ""):
                n_present += 1

    # Ensure n_total matches expected panel size (warn if not)
    if n_total != TOTAL_SNPS:
        log.warning(
            "Dosage file has %d rows; expected %d (full HIrisPlex-S panel).",
            n_total, TOTAL_SNPS,
        )
    return n_present, n_total


# ---------------------------------------------------------------------------
# Confidence label logic
# ---------------------------------------------------------------------------

def _label(
    threshold_met: bool,
    uncertainty_score: float,
    snp_coverage_pct: float,
    population_bias_warning: bool,
) -> str:
    """
    Assign HIGH / MEDIUM / LOW confidence label.

    Rules (applied in priority order):
      HIGH   : threshold met AND uncertainty ≤ 0.30 AND coverage ≥ 80%
                AND no population bias warning
      MEDIUM : threshold met AND uncertainty ≤ 0.50 AND coverage ≥ 70%
      LOW    : anything else
    """
    if (
        threshold_met
        and uncertainty_score <= UNCERTAINTY_CAP_HIGH
        and snp_coverage_pct  >= MIN_COVERAGE_HIGH
        and not population_bias_warning
    ):
        return "HIGH"
    if (
        threshold_met
        and uncertainty_score <= UNCERTAINTY_CAP_MEDIUM
        and snp_coverage_pct  >= MIN_COVERAGE_MEDIUM
    ):
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Per-trait uncertainty block
# ---------------------------------------------------------------------------

def _build_trait_report(
    trait: str,
    pred: dict,
    snp_coverage_pct: float,
    n_present: int,
    n_total: int,
    ani_proportion: float | None,
    always_bias: bool,
) -> dict:
    """Build the complete uncertainty block for one trait."""

    probabilities: dict[str, float] = pred.get("probabilities", {})
    predicted: str                  = pred.get("predicted", "unknown")
    n_imputed: int                  = pred.get("n_imputed", 0)
    imputed_rsids: list             = pred.get("imputed_rsids", [])
    model_warnings: list            = pred.get("warnings", [])

    # --- core metrics ---
    max_prob          = max(probabilities.values()) if probabilities else 0.0
    uncertainty_score = round(1.0 - max_prob, 6)
    threshold         = CALIBRATED_THRESHOLDS.get(trait, 0.70)
    threshold_met     = max_prob >= threshold

    # --- population bias flag ---
    ani_bias = (ani_proportion is not None) and (ani_proportion < 0.5)
    population_bias_warning = always_bias or ani_bias
    bias_reasons: list[str] = []
    if always_bias:
        bias_reasons.append(
            "HIrisPlex-S skin model trained predominantly on European samples; "
            "performance is reduced for South Asian subjects."
        )
    if ani_bias:
        bias_reasons.append(
            f"Subject ANI proportion {ani_proportion:.2f} < 0.50 — ancestry "
            "skews toward Ancestral South Indian, which is more distant from "
            "the model training population."
        )

    # --- confidence label ---
    confidence_label = _label(
        threshold_met=threshold_met,
        uncertainty_score=uncertainty_score,
        snp_coverage_pct=snp_coverage_pct,
        population_bias_warning=population_bias_warning,
    )

    # --- sort probabilities descending for readability ---
    sorted_probs = {
        cat: round(p, 6)
        for cat, p in sorted(probabilities.items(), key=lambda x: -x[1])
    }

    return {
        "predicted":               predicted,
        "probabilities":           sorted_probs,
        "max_probability":         round(max_prob, 6),
        "uncertainty_score":       uncertainty_score,
        "calibrated_threshold":    threshold,
        "threshold_met":           threshold_met,
        "snp_coverage_pct":        round(snp_coverage_pct, 2),
        "n_present":               n_present,
        "n_imputed":               n_imputed,
        "n_total":                 n_total,
        "imputed_rsids":           imputed_rsids,
        "population_bias_warning": population_bias_warning,
        "bias_reasons":            bias_reasons,
        "confidence_label":        confidence_label,
        "model_warnings":          model_warnings,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_report(
    predictions_path: Path,
    ancestry_path:    Path,
    dosages_path:     Path,
    output_path:      Path,
    ani_threshold:    float = 0.5,
) -> dict:
    """
    Core logic: load inputs, compute per-trait uncertainty blocks, save JSON.
    Returns the final report dict.
    """

    # ---- load inputs --------------------------------------------------------
    preds     = load_predictions(predictions_path)
    ancestry  = load_ancestry(ancestry_path)
    n_present, n_total = load_dosage_coverage(dosages_path)

    snp_coverage_pct = (n_present / n_total * 100) if n_total > 0 else 0.0
    ani_proportion: float | None = (
        ancestry.get("ani_proportion") if ancestry else None
    )

    log.info(
        "SNP coverage: %d/%d (%.1f%%)  |  ANI proportion: %s",
        n_present, n_total, snp_coverage_pct,
        f"{ani_proportion:.2f}" if ani_proportion is not None else "unknown",
    )

    # ---- per-trait report ---------------------------------------------------
    trait_reports: dict[str, dict] = {}
    all_predictions: dict = preds.get("predictions", {})

    TRAITS = ("eye", "hair", "skin")
    for trait in TRAITS:
        pred = all_predictions.get(trait, {})

        # error passthrough
        if "error" in pred:
            trait_reports[trait] = {
                "error": pred["error"],
                "confidence_label": "LOW",
                "population_bias_warning": trait == "skin",
            }
            log.warning("Trait '%s' has a prediction error: %s", trait, pred["error"])
            continue

        always_bias = (trait == "skin")   # skin always gets bias flag
        report = _build_trait_report(
            trait=trait,
            pred=pred,
            snp_coverage_pct=snp_coverage_pct,
            n_present=n_present,
            n_total=n_total,
            ani_proportion=ani_proportion,
            always_bias=always_bias,
        )
        trait_reports[trait] = report

        log.info(
            "  %-4s → predicted=%-18s  uncertainty=%.3f  "
            "threshold_met=%-5s  label=%s  bias_warning=%s",
            trait.upper(),
            report["predicted"],
            report["uncertainty_score"],
            str(report["threshold_met"]),
            report["confidence_label"],
            report["population_bias_warning"],
        )

    # ---- assemble full report -----------------------------------------------
    subject_id = (
        ancestry.get("subject_id") if ancestry else
        preds.get("subject_id", "unknown")
    )

    full_report = {
        "pipeline":          "HIrisPlex-S Uncertainty Report",
        "generated_at":      datetime.now(timezone.utc).isoformat(),
        "subject_id":        subject_id,
        "predictions_file":  str(predictions_path),
        "ancestry_file":     str(ancestry_path) if ancestry else None,
        "dosages_file":      str(dosages_path),
        "snp_coverage": {
            "n_present":          n_present,
            "n_total":            n_total,
            "coverage_pct":       round(snp_coverage_pct, 2),
        },
        "ani_proportion":    ani_proportion,
        "ani_threshold_used": ani_threshold,
        "calibrated_thresholds": CALIBRATED_THRESHOLDS,
        "trait_reports":     trait_reports,
    }

    # ---- write output -------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(full_report, fh, indent=2)
    log.info("Pigmentation report written → %s", output_path)

    return full_report


def _print_banner(report: dict) -> None:
    """Pretty-print a summary banner to stdout."""
    print("\n" + "=" * 70)
    print("  HIrisPlex-S Uncertainty Report")
    print("=" * 70)
    cov = report["snp_coverage"]
    print(f"  SNP coverage  : {cov['n_present']}/{cov['n_total']}  "
          f"({cov['coverage_pct']:.1f}%)")
    ani = report.get("ani_proportion")
    print(f"  ANI proportion: {ani:.2f}" if ani is not None
          else "  ANI proportion: unknown")
    print("-" * 70)

    LABEL_ICONS = {"HIGH": "✓", "MEDIUM": "~", "LOW": "⚠"}
    for trait in ("eye", "hair", "skin"):
        tr = report["trait_reports"].get(trait, {})
        if "error" in tr:
            print(f"  {trait.upper():<6} : ERROR — {tr['error']}")
            continue
        label  = tr["confidence_label"]
        icon   = LABEL_ICONS.get(label, "?")
        bias   = "⚡ BIAS" if tr["population_bias_warning"] else ""
        print(
            f"  {trait.upper():<6} : {tr['predicted']:<20}  "
            f"[{icon} {label:<6}]  "
            f"uncertainty={tr['uncertainty_score']:.3f}  "
            f"threshold={'✓' if tr['threshold_met'] else '✗'}  {bias}"
        )

    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute HIrisPlex-S prediction uncertainty and confidence labels."
    )
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
        help="Path to hirisplex_predictions.json  (default: %(default)s).",
    )
    parser.add_argument(
        "--ancestry",
        default=str(DEFAULT_ANCESTRY),
        help="Path to ancestry_stage2.json  (default: %(default)s).",
    )
    parser.add_argument(
        "--dosages",
        default=str(DEFAULT_DOSAGES),
        help="Path to hirisplex_dosages.csv  (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Output path for pigmentation_report.json  (default: %(default)s).",
    )
    parser.add_argument(
        "--ani-threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="ANI proportion below which population_bias_warning is set  "
             "(default: %(default)s).",
    )
    args = parser.parse_args()

    try:
        report = build_report(
            predictions_path=Path(args.predictions),
            ancestry_path=Path(args.ancestry),
            dosages_path=Path(args.dosages),
            output_path=Path(args.output),
            ani_threshold=args.ani_threshold,
        )
    except FileNotFoundError as exc:
        log.error("%s", exc)
        sys.exit(1)

    _print_banner(report)


if __name__ == "__main__":
    main()

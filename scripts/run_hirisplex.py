#!/usr/bin/env python3
"""
run_hirisplex.py
================
End-to-end driver: load hirisplex_dosages.csv → run HIrisPlex-S models →
save outputs/hirisplex_predictions.json.

Usage
-----
    python scripts/run_hirisplex.py
    python scripts/run_hirisplex.py --dosages path/to/hirisplex_dosages.csv
    python scripts/run_hirisplex.py --output  path/to/hirisplex_predictions.json
    python scripts/run_hirisplex.py --skip-validation   # skip TODO check (dev only)
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
# Ensure 'scripts/' is importable whether run as a script or via `python -m`
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.hirisplex_model import HIriPlexS  # noqa: E402

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_DOSAGES = PROJECT_ROOT / "outputs" / "hirisplex_dosages.csv"
DEFAULT_OUTPUT  = PROJECT_ROOT / "outputs" / "hirisplex_predictions.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_hirisplex")


# ---------------------------------------------------------------------------
# Dosage loader
# ---------------------------------------------------------------------------

def load_dosages(csv_path: Path) -> dict[str, str | int | float]:
    """
    Read hirisplex_dosages.csv and return a plain dict:
        rsID → dosage value  ('NA' or 0/1/2 as int)

    The CSV must have at minimum the columns 'rsID' and 'dosage'.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dosages file not found: {csv_path}")

    dosages: dict[str, str | int | float] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
            rsid    = cleaned.get("rsID", "").strip()
            raw     = cleaned.get("dosage", "NA").strip().upper()
            if not rsid:
                continue
            if raw in ("NA", "NAN", "NONE", ""):
                dosages[rsid] = "NA"
            else:
                try:
                    dosages[rsid] = int(raw)
                except ValueError:
                    try:
                        dosages[rsid] = float(raw)
                    except ValueError:
                        log.warning("Could not parse dosage '%s' for %s; treating as NA.", raw, rsid)
                        dosages[rsid] = "NA"

    present = sum(1 for v in dosages.values() if v != "NA")
    log.info("Loaded dosages for %d SNPs (%d present, %d NA).",
             len(dosages), present, len(dosages) - present)
    return dosages


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def _fmt_probs(prob_dict: dict[str, float]) -> str:
    parts = [f"{cat}: {p*100:.1f}%" for cat, p in
             sorted(prob_dict.items(), key=lambda x: -x[1])]
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run HIrisPlex-S pigmentation predictions from dosage CSV."
    )
    parser.add_argument(
        "--dosages",
        default=str(DEFAULT_DOSAGES),
        help="Path to hirisplex_dosages.csv  (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the output predictions JSON  (default: %(default)s).",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.50,
        metavar="PROB",
        help="Minimum winning-class probability to flag as high-confidence  "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip coefficient TODO-placeholder validation  (development only).",
    )
    args = parser.parse_args()

    # ---- load dosages -------------------------------------------------------
    dosages = load_dosages(Path(args.dosages))

    # ---- initialise model ---------------------------------------------------
    model = HIriPlexS(confidence_threshold=args.confidence_threshold)

    # ---- validate coefficients (unless explicitly skipped) ------------------
    if not args.skip_validation:
        log.info("Validating coefficients …")
        try:
            model.validate()
        except ValueError as exc:
            log.error("Coefficient validation FAILED:\n%s", exc)
            log.error(
                "Fill in hirisplex_coefficients.json with the published betas "
                "from Walsh et al. 2017 (eye/hair) and Chaitanya et al. 2018 (skin), "
                "then re-run."
            )
            sys.exit(1)
    else:
        log.warning(
            "--skip-validation is active: coefficient TODO-check is disabled. "
            "This is for development only; do NOT use in production."
        )

    # ---- run predictions ----------------------------------------------------
    results: dict[str, dict] = {}
    errors:  list[str]       = []

    for trait, method in (
        ("eye",  model.predict_eye),
        ("hair", model.predict_hair),
        ("skin", model.predict_skin),
    ):
        log.info("Running %s model …", trait)
        try:
            out = method(dosages)
            results[trait] = out
            probs_str = _fmt_probs(out["probabilities"])
            conf_str  = "HIGH" if out["high_confidence"] else "LOW"
            log.info(
                "  %-4s → predicted=%-15s  confidence=%-4s  "
                "present=%d  imputed=%d",
                trait.upper(), out["predicted"], conf_str,
                out["n_present"], out["n_imputed"],
            )
            log.info("       %s", probs_str)
            if out["warnings"]:
                for w in out["warnings"]:
                    log.warning("  [%s] %s", trait, w)
        except ValueError as exc:
            log.error("Prediction failed for %s model: %s", trait, exc)
            errors.append(f"{trait}: {exc}")
            results[trait] = {"error": str(exc)}

    # ---- assemble output document -------------------------------------------
    output_doc = {
        "pipeline": "HIrisPlex-S (Chaitanya et al. 2018 / Walsh et al. 2017)",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dosages_file": str(args.dosages),
        "confidence_threshold": args.confidence_threshold,
        "predictions": results,
    }

    # ---- write JSON ---------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(output_doc, fh, indent=2)
    log.info("Predictions written → %s", out_path)

    # ---- summary banner -------------------------------------------------------
    print("\n" + "=" * 60)
    print("  HIrisPlex-S Prediction Summary")
    print("=" * 60)
    for trait in ("eye", "hair", "skin"):
        res = results.get(trait, {})
        if "error" in res:
            print(f"  {trait.upper():<6} : ERROR — {res['error']}")
        else:
            prob_str = "  ".join(
                f"{c}={p*100:.1f}%"
                for c, p in sorted(res["probabilities"].items(),
                                    key=lambda x: -x[1])
            )
            conf = "✓ high" if res["high_confidence"] else "⚠ low"
            print(f"  {trait.upper():<6} : {res['predicted']:<18} "
                  f"[conf={conf}]  |  {prob_str}")
    print("=" * 60 + "\n")

    if errors:
        log.error("%d model(s) failed to produce predictions.", len(errors))
        sys.exit(1)


if __name__ == "__main__":
    main()

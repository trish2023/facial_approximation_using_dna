"""build_hirisplex_reference.py — build HIrisPlex-S 41-SNP reference and coefficient scaffold."""

import argparse
import csv
import io
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIRISPLEX_CSV_URL = "https://hirisplex.erasmusmc.nl/examples/HIrisPlex.csv"
CONNECT_TIMEOUT = 30
READ_TIMEOUT = 120
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2
EXPECTED_SNP_COUNT = 41

MISSING_WARN_FRACTION = 0.10
MISSING_FAIL_FRACTION = 0.20

# ---------------------------------------------------------------------------
# HIrisPlex-S SNP metadata
# Source: Walsh et al. 2017, Forensic Science International: Genetics
# doi:10.1016/j.fsigen.2017.04.018 — Table 1 and Supplementary Tables.
# Chromosomes are GRCh37/38-consistent (chromosome number only).
# ---------------------------------------------------------------------------
SNP_METADATA: Dict[str, Dict[str, str]] = {
    "rs312262906":  {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs11547464":   {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs885479":     {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs1805008":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs1805005":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs1805006":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs1805007":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs1805009":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs201326893_Y152OCH": {"gene": "MC1R",  "chromosome": "16", "trait_model": "hair+skin"},
    "rs2228479":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs1110400":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "hair+skin"},
    "rs28777":      {"gene": "SLC45A2",      "chromosome": "5",  "trait_model": "hair+skin"},
    "rs16891982":   {"gene": "SLC45A2",      "chromosome": "5",  "trait_model": "eye+hair+skin"},
    "rs12821256":   {"gene": "KITLG",        "chromosome": "12", "trait_model": "hair"},
    "rs4959270":    {"gene": "EXOC2",        "chromosome": "6",  "trait_model": "hair"},
    "rs12203592":   {"gene": "IRF4",         "chromosome": "6",  "trait_model": "hair+skin"},
    "rs1042602":    {"gene": "TYR",          "chromosome": "11", "trait_model": "hair+skin"},
    "rs1800407":    {"gene": "OCA2",         "chromosome": "15", "trait_model": "eye"},
    "rs2402130":    {"gene": "SLC24A4",      "chromosome": "14", "trait_model": "eye+hair"},
    "rs12913832":   {"gene": "HERC2",        "chromosome": "15", "trait_model": "eye+hair+skin"},
    "rs2378249":    {"gene": "PIGU/ASIP",    "chromosome": "20", "trait_model": "hair"},
    "rs12896399":   {"gene": "SLC24A4",      "chromosome": "14", "trait_model": "eye+hair"},
    "rs1393350":    {"gene": "TYR",          "chromosome": "11", "trait_model": "eye+hair"},
    "rs683":        {"gene": "TYRP1",        "chromosome": "9",  "trait_model": "hair"},
    "rs3114908":    {"gene": "ASIP",         "chromosome": "20", "trait_model": "skin"},
    "rs1800414":    {"gene": "OCA2",         "chromosome": "15", "trait_model": "skin"},
    "rs10756819":   {"gene": "BNC2",         "chromosome": "9",  "trait_model": "skin"},
    "rs2238289":    {"gene": "HERC2",        "chromosome": "15", "trait_model": "skin"},
    "rs17128291":   {"gene": "HERC2",        "chromosome": "15", "trait_model": "skin"},
    "rs6497292":    {"gene": "HERC2",        "chromosome": "15", "trait_model": "skin"},
    "rs1129038":    {"gene": "HERC2",        "chromosome": "15", "trait_model": "skin"},
    "rs1667394":    {"gene": "HERC2",        "chromosome": "15", "trait_model": "skin"},
    "rs1126809":    {"gene": "TYR",          "chromosome": "11", "trait_model": "skin"},
    "rs1470608":    {"gene": "OCA2",         "chromosome": "15", "trait_model": "skin"},
    "rs1545397":    {"gene": "OCA2",         "chromosome": "15", "trait_model": "skin"},
    "rs6119471":    {"gene": "ASIP",         "chromosome": "20", "trait_model": "skin"},
    "rs6059655":    {"gene": "RALY/ASIP",    "chromosome": "20", "trait_model": "skin"},
    "rs12441727":   {"gene": "OCA2",         "chromosome": "15", "trait_model": "skin"},
    "rs3212355":    {"gene": "MC1R",         "chromosome": "16", "trait_model": "skin"},
    "rs8051733":    {"gene": "DEF8/MC1R",    "chromosome": "16", "trait_model": "skin"},
    "rs2504853":    {"gene": "UGT1A",        "chromosome": "2",  "trait_model": "skin"},
}

# ---------------------------------------------------------------------------
# Model SNP orderings (Walsh et al. 2017, Supplementary Tables)
# ---------------------------------------------------------------------------
EYE_MODEL_SNPS: List[str] = [
    "rs12913832",
    "rs1800407",
    "rs12896399",
    "rs16891982",
    "rs1393350",
    "rs12203592",
]

HAIR_MODEL_SNPS: List[str] = [
    "rs312262906",
    "rs11547464",
    "rs885479",
    "rs1805008",
    "rs1805005",
    "rs1805006",
    "rs1805007",
    "rs1805009",
    "rs201326893_Y152OCH",
    "rs2228479",
    "rs1110400",
    "rs28777",
    "rs16891982",
    "rs12821256",
    "rs4959270",
    "rs12203592",
    "rs1042602",
    "rs2402130",
    "rs12913832",
    "rs2378249",
    "rs12896399",
    "rs1393350",
    "rs683",
]

SKIN_MODEL_SNPS: List[str] = [
    "rs312262906",
    "rs11547464",
    "rs885479",
    "rs1805008",
    "rs1805005",
    "rs1805006",
    "rs1805007",
    "rs1805009",
    "rs201326893_Y152OCH",
    "rs2228479",
    "rs1110400",
    "rs28777",
    "rs16891982",
    "rs12203592",
    "rs1042602",
    "rs12913832",
    "rs3114908",
    "rs1800414",
    "rs10756819",
    "rs2238289",
    "rs17128291",
    "rs6497292",
    "rs1129038",
    "rs1667394",
    "rs1126809",
    "rs1470608",
    "rs1545397",
    "rs6119471",
    "rs6059655",
    "rs12441727",
    "rs3212355",
    "rs8051733",
    "rs2504853",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def resolve_paths(config: dict) -> Tuple[Path, Path, Path]:
    """Return (project_root, snp_csv_path, coefficients_json_path)."""
    root = Path(__file__).resolve().parents[1]
    snp_csv = root / config["paths"]["hirisplex_snp_list"]
    coeffs_json = snp_csv.parent / "hirisplex_coefficients.json"
    return root, snp_csv, coeffs_json


# ---------------------------------------------------------------------------
# Download + parse HIrisPlex-S example CSV
# ---------------------------------------------------------------------------

def download_hirisplex_csv(url: str, session: requests.Session) -> Optional[str]:
    """Fetch the HIrisPlex example CSV with retries; returns decoded text or None on failure."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT))
            resp.raise_for_status()
            log.info("Downloaded %s (%d bytes).", url, len(resp.content))
            return resp.text

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            log.warning("HTTP %s from %s — will use hardcoded SNP list.", status, url)
            return None

        except (requests.ConnectionError, requests.Timeout) as exc:
            log.warning("Network error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.info("Retrying in %s seconds...", wait)
            time.sleep(wait)

    log.warning("Download failed after %d attempts — using hardcoded SNP list.", MAX_RETRIES)
    return None


def parse_snp_headers(csv_text: str) -> List[str]:
    """Extract the rsID column headers from the HIrisPlex example CSV."""
    reader = csv.reader(io.StringIO(csv_text))
    try:
        header = next(reader)
    except StopIteration:
        log.error("HIrisPlex CSV is empty; no header row found.")
        sys.exit(1)

    # The example file typically has one leading ID column followed by 41 rsIDs.
    # Filter to columns that look like rsIDs (or the Y152OCH special token).
    snps: List[str] = []
    for col in header:
        name = col.strip()
        if not name:
            continue
        if name.lower().startswith("rs") or "Y152OCH" in name:
            snps.append(name)

    if len(snps) != EXPECTED_SNP_COUNT:
        log.warning(
            "Parsed %d SNP headers from CSV; expected %d. Proceeding anyway.",
            len(snps),
            EXPECTED_SNP_COUNT,
        )
    else:
        log.info("Parsed %d SNP headers from HIrisPlex CSV.", len(snps))
    return snps


def write_snp_csv(snps: List[str], out_path: Path) -> None:
    """Write the 41-SNP reference CSV (rsID, gene, chromosome, trait_model)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    missing_meta: List[str] = []
    with open(out_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rsID", "gene", "chromosome", "trait_model"])
        for rsid in snps:
            meta = SNP_METADATA.get(rsid)
            if meta is None:
                missing_meta.append(rsid)
                writer.writerow([rsid, "UNKNOWN", "UNKNOWN", "UNKNOWN"])
                continue
            writer.writerow([rsid, meta["gene"], meta["chromosome"], meta["trait_model"]])

    if missing_meta:
        log.warning(
            "No hardcoded metadata for %d rsIDs (written as UNKNOWN): %s",
            len(missing_meta),
            ", ".join(missing_meta),
        )
    log.info("Wrote SNP reference CSV: %s", out_path)


# ---------------------------------------------------------------------------
# Coefficients scaffold
# ---------------------------------------------------------------------------

def build_coefficients_scaffold() -> Dict[str, Any]:
    """Build the placeholder coefficients JSON with TODO values."""

    def todo_vec(n: int) -> List[str]:
        return ["TODO"] * n

    eye_cats = ["blue", "intermediate", "brown"]
    hair_cats = ["blond", "brown", "black", "red"]
    skin_cats = ["very_pale", "pale", "intermediate", "dark", "dark_to_black"]

    scaffold: Dict[str, Any] = {
        "source": "Walsh et al. 2017 doi:10.1016/j.fsigen.2017.04.018",
        "note": "Replace TODO arrays with values from Supplementary Tables in the paper",
        "eye_model": {
            "categories": eye_cats,
            "intercepts": {c: "TODO" for c in eye_cats},
            "snp_order": list(EYE_MODEL_SNPS),
            "betas": {c: todo_vec(len(EYE_MODEL_SNPS)) for c in eye_cats},
        },
        "hair_model": {
            "categories": hair_cats,
            "intercepts": {c: "TODO" for c in hair_cats},
            "snp_order": list(HAIR_MODEL_SNPS),
            "betas": {c: todo_vec(len(HAIR_MODEL_SNPS)) for c in hair_cats},
        },
        "skin_model": {
            "categories": skin_cats,
            "intercepts": {c: "TODO" for c in skin_cats},
            "snp_order": list(SKIN_MODEL_SNPS),
            "betas": {c: todo_vec(len(SKIN_MODEL_SNPS)) for c in skin_cats},
        },
    }
    return scaffold


def write_coefficients_json(scaffold: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(scaffold, fh, indent=2)
    log.info("Wrote coefficients scaffold JSON: %s", out_path)


# ---------------------------------------------------------------------------
# Dosage vector validation
# ---------------------------------------------------------------------------

def _is_missing(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def validate_dosage_vector(
    dosages: List[Optional[Union[int, float]]],
    snp_order: List[str],
) -> Dict[str, Any]:
    """Validate a dosage vector against the expected SNP order and dosage encoding."""
    errors: List[str] = []
    warnings: List[str] = []

    if len(dosages) != len(snp_order):
        errors.append(
            f"Length mismatch: got {len(dosages)} dosages, expected {len(snp_order)}."
        )

    missing_count = 0
    for idx, val in enumerate(dosages):
        rsid = snp_order[idx] if idx < len(snp_order) else f"<index {idx}>"
        if _is_missing(val):
            missing_count += 1
            continue
        # Treat floats that are exactly 0/1/2 as valid (eases numpy input)
        if isinstance(val, float) and not math.isnan(val) and val.is_integer():
            val_int: Optional[int] = int(val)
        elif isinstance(val, int) and not isinstance(val, bool):
            val_int = val
        else:
            errors.append(
                f"{rsid} (index {idx}): non-numeric dosage value {val!r}."
            )
            continue
        if val_int not in (0, 1, 2):
            errors.append(
                f"{rsid} (index {idx}): dosage {val_int} outside {{0,1,2}}."
            )

    total = len(dosages) if dosages else 0
    missing_fraction = (missing_count / total) if total else 0.0

    if missing_fraction > MISSING_FAIL_FRACTION:
        errors.append(
            f"Missing fraction {missing_fraction:.1%} exceeds hard limit "
            f"{MISSING_FAIL_FRACTION:.0%}."
        )
    elif missing_fraction >= MISSING_WARN_FRACTION:
        warnings.append(
            f"Missing fraction {missing_fraction:.1%} is between "
            f"{MISSING_WARN_FRACTION:.0%} and {MISSING_FAIL_FRACTION:.0%}; "
            f"predictions may be unreliable."
        )

    return {
        "valid": not errors,
        "errors": errors,
        "warnings": warnings,
        "missing_count": missing_count,
        "missing_fraction": missing_fraction,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the HIrisPlex-S 41-SNP reference and coefficients scaffold."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--url",
        default=HIRISPLEX_CSV_URL,
        help="Override URL for the HIrisPlex example CSV.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_config(config_path)
    _, snp_csv_path, coeffs_json_path = resolve_paths(config)

    session = requests.Session()
    session.headers.update({"User-Agent": "dna-facial-approx/1.0 (academic research)"})

    # -- Download + parse SNP list (fall back to hardcoded list on failure) --
    csv_text = download_hirisplex_csv(args.url, session)
    if csv_text is not None:
        snps = parse_snp_headers(csv_text)
    else:
        snps = list(SNP_METADATA.keys())
        log.info(
            "Using hardcoded SNP list (%d SNPs) — download unavailable.", len(snps)
        )

    # -- Write SNP reference CSV --------------------------------------------
    write_snp_csv(snps, snp_csv_path)

    # -- Write coefficients scaffold JSON -----------------------------------
    scaffold = build_coefficients_scaffold()
    write_coefficients_json(scaffold, coeffs_json_path)

    # -- Sanity-check dosage vector validator -------------------------------
    good_vec: List[Optional[Union[int, float]]] = [0] * len(EYE_MODEL_SNPS)
    good_result = validate_dosage_vector(good_vec, EYE_MODEL_SNPS)

    bad_len = len(EYE_MODEL_SNPS)
    bad_vec: List[Optional[Union[int, float]]] = [3, -1, 0] + [0] * (bad_len - 3)
    bad_result = validate_dosage_vector(bad_vec, EYE_MODEL_SNPS)

    print("\n" + "=" * 60)
    print("  HIrisPlex-S reference build summary")
    print("=" * 60)
    print(f"  SNPs parsed         : {len(snps)}")
    print(f"  SNP CSV             : {snp_csv_path}")
    print(f"  Coefficients JSON   : {coeffs_json_path}")
    print(f"  Eye model SNPs      : {len(EYE_MODEL_SNPS)}")
    print(f"  Hair model SNPs     : {len(HAIR_MODEL_SNPS)}")
    print(f"  Skin model SNPs     : {len(SKIN_MODEL_SNPS)}")
    print("-" * 60)
    print("  Dosage validation — synthetic GOOD vector (all zeros):")
    print(f"    valid           : {good_result['valid']}")
    print(f"    errors          : {good_result['errors']}")
    print(f"    warnings        : {good_result['warnings']}")
    print(f"    missing_count   : {good_result['missing_count']}")
    print(f"    missing_fraction: {good_result['missing_fraction']:.3f}")
    print("-" * 60)
    print("  Dosage validation — synthetic BAD vector ([3, -1, 0, ...]):")
    print(f"    valid           : {bad_result['valid']}")
    print(f"    errors          : {bad_result['errors']}")
    print(f"    warnings        : {bad_result['warnings']}")
    print(f"    missing_count   : {bad_result['missing_count']}")
    print(f"    missing_fraction: {bad_result['missing_fraction']:.3f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Download 1000 Genomes Phase 3 reference data for the DNA-based facial
approximation pipeline.

Downloads:
  1. Sample panel file    → data/reference/1000g_panel.txt
  2. Per-chromosome VCFs  → data/reference/1000g_vcf/
     (GRCh37, 2504 unrelated individuals, ~1 GB per chromosome)

The South Asian (SAS) super-population provides our primary Indian-context
reference (489 samples across BEB, GIH, ITU, PJL, STU).

Usage
-----
    python scripts/download_reference_data.py                     # panel + all autosomes
    python scripts/download_reference_data.py --chr 15 16         # only chr 15 & 16
    python scripts/download_reference_data.py --panel-only        # just the sample panel
    python scripts/download_reference_data.py --chr 22 --dry-run  # show URLs, don't download
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DIR = PROJECT_ROOT / "data" / "reference"
VCF_DIR = REF_DIR / "1000g_vcf"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "reference_panel_summary.json"

BASE_FTP = "https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"

PANEL_URL = f"{BASE_FTP}/integrated_call_samples_v3.20130502.ALL.panel"
PANEL_DEST = REF_DIR / "1000g_panel.txt"

VCF_TEMPLATE = (
    f"{BASE_FTP}/"
    "ALL.chr{chrom}.phase3_shapeit2_mvncall_integrated_v5b.20130502"
    ".genotypes.vcf.gz"
)
TBI_TEMPLATE = VCF_TEMPLATE + ".tbi"

AUTOSOMES = [str(c) for c in range(1, 23)]
ALL_CHROMS = AUTOSOMES + ["X"]

SAS_POPULATIONS = {
    "BEB": "Bengali in Bangladesh",
    "GIH": "Gujarati Indians in Houston, TX",
    "ITU": "Indian Telugu in the UK",
    "PJL": "Punjabi in Lahore, Pakistan",
    "STU": "Sri Lankan Tamil in the UK",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===================================================================
# Download helpers
# ===================================================================

def download_file(
    url: str,
    dest: Path,
    chunk_size: int = 1 << 20,
    desc: str | None = None,
) -> Path:
    """Download *url* → *dest* with resume support and a progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    local_size = dest.stat().st_size if dest.exists() else 0
    headers: dict[str, str] = {}
    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"

    resp = requests.get(url, headers=headers, stream=True, timeout=60)

    if resp.status_code == 416:
        log.info("Already complete: %s", dest.name)
        return dest
    resp.raise_for_status()

    is_resumed = resp.status_code == 206
    content_length = resp.headers.get("Content-Length")
    total = int(content_length) if content_length else None
    if is_resumed and total:
        total += local_size

    mode = "ab" if is_resumed else "wb"
    initial = local_size if is_resumed else 0

    with (
        open(dest, mode) as fh,
        tqdm(
            total=total,
            initial=initial,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=desc or dest.name,
            ncols=88,
        ) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))

    return dest


# ===================================================================
# Panel analysis
# ===================================================================

def download_and_analyse_panel() -> pd.DataFrame:
    """Download the sample panel and print population stats."""
    log.info("Downloading sample panel …")
    download_file(PANEL_URL, PANEL_DEST, desc="1000g_panel.txt")

    panel = pd.read_csv(PANEL_DEST, sep="\t")
    log.info("Panel loaded: %d samples", len(panel))

    sas = panel[panel["super_pop"] == "SAS"]
    log.info("South Asian (SAS) samples: %d", len(sas))
    for pop, full_name in SAS_POPULATIONS.items():
        n = len(sas[sas["pop"] == pop])
        log.info("  %-4s  %3d  %s", pop, n, full_name)

    return panel


# ===================================================================
# VCF downloads
# ===================================================================

def download_chromosome_vcf(chrom: str, dry_run: bool = False) -> None:
    """Download the VCF + index for one chromosome."""
    vcf_url = VCF_TEMPLATE.format(chrom=chrom)
    tbi_url = TBI_TEMPLATE.format(chrom=chrom)

    vcf_name = vcf_url.rsplit("/", 1)[-1]
    tbi_name = tbi_url.rsplit("/", 1)[-1]

    vcf_dest = VCF_DIR / vcf_name
    tbi_dest = VCF_DIR / tbi_name

    if dry_run:
        log.info("[dry-run] chr%s VCF: %s", chrom, vcf_url)
        log.info("[dry-run] chr%s TBI: %s", chrom, tbi_url)
        return

    log.info("Downloading chr%s …", chrom)
    download_file(tbi_url, tbi_dest, desc=f"chr{chrom}.tbi")
    download_file(vcf_url, vcf_dest, desc=f"chr{chrom}.vcf.gz")


# ===================================================================
# Summary
# ===================================================================

def save_summary(panel: pd.DataFrame, chromosomes: list[str]) -> None:
    """Write a JSON summary of what was downloaded."""
    sas = panel[panel["super_pop"] == "SAS"]
    summary = {
        "source": "1000 Genomes Phase 3 (GRCh37)",
        "base_url": BASE_FTP,
        "total_samples": len(panel),
        "super_populations": panel["super_pop"].value_counts().to_dict(),
        "sas_samples": len(sas),
        "sas_populations": sas["pop"].value_counts().to_dict(),
        "chromosomes_downloaded": chromosomes,
        "panel_file": str(PANEL_DEST),
        "vcf_directory": str(VCF_DIR),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary saved → %s", SUMMARY_PATH)


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download 1000 Genomes Phase 3 reference data.",
    )
    p.add_argument(
        "--chr",
        nargs="*",
        default=None,
        metavar="N",
        help="Chromosomes to download (e.g., 1 2 22 X). Default: all autosomes.",
    )
    p.add_argument(
        "--panel-only",
        action="store_true",
        help="Only download and analyse the sample panel (skip VCFs).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print download URLs without actually downloading.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        panel = download_and_analyse_panel()
    except requests.HTTPError as exc:
        log.error("Failed to download panel: %s", exc)
        sys.exit(1)

    if args.panel_only:
        save_summary(panel, [])
        return

    chroms = args.chr if args.chr else AUTOSOMES
    invalid = [c for c in chroms if c not in ALL_CHROMS]
    if invalid:
        log.error("Invalid chromosome(s): %s.  Valid: %s", invalid, ALL_CHROMS)
        sys.exit(1)

    VCF_DIR.mkdir(parents=True, exist_ok=True)
    for chrom in chroms:
        try:
            download_chromosome_vcf(chrom, dry_run=args.dry_run)
        except requests.HTTPError as exc:
            log.error("Failed on chr%s: %s", chrom, exc)
            continue

    save_summary(panel, chroms)
    log.info("All done.  VCFs in %s", VCF_DIR)


if __name__ == "__main__":
    main()

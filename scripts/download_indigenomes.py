#!/usr/bin/env python3
"""
Download the IndiGenomes VCF from the CSIR-IGIB ClinGen portal, validate it
with cyvcf2, compute summary statistics, and write a JSON report.

Usage
-----
    python scripts/download_indigenomes.py                        # use defaults
    python scripts/download_indigenomes.py --url <direct_url>     # override URL
    python scripts/download_indigenomes.py --local <path.vcf.gz>  # skip download, validate existing file

The download is resumable: if a partial file exists the script sends an HTTP
Range header so only the remaining bytes are fetched.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw" / "indigenomes"
SUMMARY_PATH = PROJECT_ROOT / "data" / "processed" / "indigenomes_summary.json"

INDIGEN_PORTAL = "http://clingen.igib.res.in/indigen/"
DEFAULT_FILENAME = "indigen_1029.vcf.gz"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ===================================================================
# 1. Download with resume support and progress bar
# ===================================================================

def _resolve_download_url(url: str | None) -> str:
    """Return an explicit download URL or fall back to the portal link."""
    if url:
        return url

    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as fh:
            cfg = yaml.safe_load(fh)
        configured = cfg.get("reference_panels", {}).get("indigenomes_vcf", "")
        if configured and "<PATH_TO>" not in configured and configured.startswith("http"):
            return configured

    return INDIGEN_PORTAL


def _get_remote_size(url: str, session: requests.Session) -> int | None:
    """HEAD request to learn the total file size (None if server won't say)."""
    try:
        resp = session.head(url, allow_redirects=True, timeout=30)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except requests.RequestException:
        return None


def download_vcf(url: str, dest: Path, chunk_size: int = 1 << 20) -> Path:
    """
    Download *url* to *dest*, resuming from where a previous attempt left off.

    Returns the Path to the downloaded file.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "IndiGenomes-Downloader/1.0"})

    remote_size = _get_remote_size(url, session)
    local_size = dest.stat().st_size if dest.exists() else 0

    if remote_size and local_size >= remote_size:
        log.info("File already fully downloaded (%s bytes). Skipping.", local_size)
        return dest

    headers: dict[str, str] = {}
    if local_size > 0:
        headers["Range"] = f"bytes={local_size}-"
        log.info("Resuming download from byte %s …", local_size)

    try:
        resp = session.get(url, headers=headers, stream=True, timeout=60)

        if resp.status_code == 416:
            log.info("Server says range not satisfiable — file is complete.")
            return dest

        resp.raise_for_status()
    except requests.ConnectionError as exc:
        log.error("Connection failed: %s", exc)
        log.error(
            "The IndiGenomes portal may require manual access approval.\n"
            "  1. Visit %s in a browser.\n"
            "  2. Download the VCF manually to %s\n"
            "  3. Re-run with:  python %s --local %s",
            INDIGEN_PORTAL, dest, __file__, dest,
        )
        sys.exit(1)
    except requests.HTTPError as exc:
        log.error("HTTP error: %s", exc)
        sys.exit(1)

    is_resumed = resp.status_code == 206
    total = remote_size or int(resp.headers.get("Content-Length", 0)) or None
    if is_resumed and total:
        total = local_size + (total - local_size) if total > local_size else total

    mode = "ab" if is_resumed else "wb"
    initial = local_size if is_resumed else 0

    log.info(
        "Downloading %s → %s  (%s)",
        url,
        dest,
        f"{total / 1e6:.1f} MB" if total else "unknown size",
    )

    with (
        open(dest, mode) as fh,
        tqdm(
            total=total,
            initial=initial,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest.name,
            ncols=88,
        ) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                bar.update(len(chunk))

    log.info("Download complete: %s", dest)
    return dest


# ===================================================================
# 2. Validate that the file is a gzipped VCF (read first header lines)
# ===================================================================

def validate_vcf_header(vcf_path: Path, n_header_lines: int = 5) -> list[str]:
    """
    Open *vcf_path* with cyvcf2, read the raw header, and return the first
    *n_header_lines* meta-information lines.  Raises on any parse error.
    """
    try:
        from cyvcf2 import VCF
    except ImportError:
        log.warning("cyvcf2 not installed — falling back to gzip header check.")
        return _validate_gzip_fallback(vcf_path, n_header_lines)

    log.info("Validating VCF header with cyvcf2 …")
    vcf = VCF(str(vcf_path))
    raw_header: str = vcf.raw_header
    vcf.close()

    lines = [l for l in raw_header.splitlines() if l.startswith("##")]
    first_n = lines[:n_header_lines]

    if not first_n or not first_n[0].startswith("##fileformat=VCF"):
        raise ValueError(
            f"File does not start with a valid VCF header: {vcf_path}"
        )

    for line in first_n:
        log.info("  %s", line)

    log.info("VCF header validated (%d meta-information lines total).", len(lines))
    return first_n


def _validate_gzip_fallback(vcf_path: Path, n: int) -> list[str]:
    """Minimal gzip-based header check when cyvcf2 is unavailable."""
    lines: list[str] = []
    with gzip.open(vcf_path, "rt") as fh:
        for raw_line in fh:
            if raw_line.startswith("##"):
                lines.append(raw_line.rstrip())
                if len(lines) >= n:
                    break
            else:
                break

    if not lines or not lines[0].startswith("##fileformat=VCF"):
        raise ValueError(
            f"File does not look like a gzipped VCF: {vcf_path}"
        )

    for line in lines:
        log.info("  %s", line)
    return lines


# ===================================================================
# 3. Compute summary statistics
# ===================================================================

def compute_summary(vcf_path: Path) -> dict:
    """
    Iterate the VCF once and collect:
      - total variant count
      - chromosome distribution
      - whether AF is present in INFO
    """
    try:
        from cyvcf2 import VCF
    except ImportError:
        log.error("cyvcf2 is required for summary statistics. Install it and retry.")
        sys.exit(1)

    log.info("Scanning variants (this may take a while for whole-genome data) …")
    vcf = VCF(str(vcf_path))

    chrom_counts: Counter[str] = Counter()
    total_variants = 0
    af_present_count = 0
    af_checked = False
    has_af = False
    sample_count = len(vcf.samples)

    t0 = time.monotonic()
    for variant in vcf:
        total_variants += 1
        chrom_counts[variant.CHROM] += 1

        if not af_checked:
            af_val = variant.INFO.get("AF")
            if af_val is not None:
                has_af = True
            af_checked = True

        if total_variants <= 1000:
            af_val = variant.INFO.get("AF")
            if af_val is not None:
                af_present_count += 1

        if total_variants % 5_000_000 == 0:
            elapsed = time.monotonic() - t0
            log.info("  … %d M variants scanned (%.0f s)", total_variants // 1_000_000, elapsed)

    vcf.close()
    elapsed = time.monotonic() - t0

    chrom_sorted = dict(
        sorted(chrom_counts.items(), key=_chrom_sort_key)
    )

    af_note = (
        f"AF found in INFO for {af_present_count}/1000 sampled variants"
        if total_variants >= 1000
        else f"AF found in INFO for {af_present_count}/{total_variants} variants"
    )

    summary = {
        "file": str(vcf_path),
        "samples": sample_count,
        "total_variants": total_variants,
        "chromosomes": chrom_sorted,
        "has_af_annotation": has_af,
        "af_detail": af_note,
        "scan_time_seconds": round(elapsed, 1),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }

    log.info("Scan complete in %.1f s", elapsed)
    log.info("  Samples:          %d", sample_count)
    log.info("  Total variants:   %s", f"{total_variants:,}")
    log.info("  Chromosomes:      %d", len(chrom_sorted))
    log.info("  AF in INFO:       %s", "yes" if has_af else "no")

    for chrom, count in chrom_sorted.items():
        log.info("    %-6s %s", chrom, f"{count:>12,}")

    return summary


def _chrom_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    """Sort chromosomes numerically (1–22) then X, Y, MT, then everything else."""
    chrom = item[0].replace("chr", "")
    if chrom.isdigit():
        return (0, chrom.zfill(2))
    order = {"X": 23, "Y": 24, "MT": 25, "M": 25}
    return (1, str(order.get(chrom, 99)) + chrom)


# ===================================================================
# 4. Save summary JSON
# ===================================================================

def save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary saved → %s", path)


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and summarise the IndiGenomes VCF.",
    )
    p.add_argument(
        "--url",
        default=None,
        help=(
            "Direct download URL for the VCF. "
            "If omitted, the script tries config.yaml then falls back to the portal page."
        ),
    )
    p.add_argument(
        "--local",
        default=None,
        type=Path,
        help="Path to an already-downloaded VCF. Skips the download step.",
    )
    p.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR,
        type=Path,
        help=f"Directory for the downloaded file (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--filename",
        default=DEFAULT_FILENAME,
        help=f"Filename for the downloaded VCF (default: {DEFAULT_FILENAME}).",
    )
    p.add_argument(
        "--summary-path",
        default=SUMMARY_PATH,
        type=Path,
        help=f"Where to write the JSON summary (default: {SUMMARY_PATH}).",
    )
    p.add_argument(
        "--skip-scan",
        action="store_true",
        help="Only download and validate; skip the full variant scan.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- Step 1: Obtain the VCF file ----------------------------------------
    if args.local:
        vcf_path = args.local.resolve()
        if not vcf_path.exists():
            log.error("Local file not found: %s", vcf_path)
            sys.exit(1)
        log.info("Using local file: %s", vcf_path)
    else:
        url = _resolve_download_url(args.url)
        dest = args.out_dir / args.filename
        log.info("Download URL resolved to: %s", url)
        vcf_path = download_vcf(url, dest)

    # --- Step 2: Validate header --------------------------------------------
    try:
        validate_vcf_header(vcf_path)
    except (ValueError, OSError) as exc:
        log.error("Validation failed: %s", exc)
        sys.exit(1)

    # --- Step 3 & 4: Scan and save summary ----------------------------------
    if args.skip_scan:
        log.info("--skip-scan set; skipping full variant scan.")
        return

    summary = compute_summary(vcf_path)
    save_summary(summary, args.summary_path)

    log.info("Done.")


if __name__ == "__main__":
    main()

"""
download_indigenomes.py

Downloads the IndiGenomes population VCF, validates it, and writes a summary
JSON to data/processed/indigenomes_summary.json.

Usage
-----
    python scripts/download_indigenomes.py --url <VCF_URL> [--config config.yaml]

The --url argument must be the direct link to the .vcf.gz file provided to
you by IGIB after your data-access request is approved.  The portal page
http://clingen.igib.res.in/indigen/ requires institutional login; this script
handles the download once you have an authenticated URL or a local copy.

Resume behaviour
----------------
If the destination file already exists and the server supports HTTP Range
requests, the download is resumed from the byte offset already on disk.
If the file is already fully downloaded (Content-Length matches), the
download step is skipped entirely and validation runs on the existing file.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional, Tuple

import requests
import yaml
from tqdm import tqdm

# cyvcf2 is imported lazily after the download so import errors surface clearly
# at validation time rather than before the download starts.

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
CHUNK_SIZE = 1 * 1024 * 1024          # 1 MiB read/write chunks
CONNECT_TIMEOUT = 30                   # seconds
READ_TIMEOUT = 120                     # seconds — large genomic files are slow
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2                 # exponential back-off: 2, 4, 8 … seconds
HEADER_LINES_TO_VALIDATE = 5
VARIANT_COUNT_LIMIT = 5_000_000        # safety: stop counting after N variants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def resolve_paths(config: dict) -> Tuple[Path, Path]:
    """Return (dest_dir, summary_json_path) from config."""
    root = Path(__file__).resolve().parents[1]
    vcf_path = root / config["paths"]["indigenomes_vcf"]
    summary_path = root / config["paths"]["processed_dir"] / "indigenomes_summary.json"
    return vcf_path, summary_path


def remote_file_size(url: str, session: requests.Session) -> Optional[int]:
    """HEAD request to retrieve Content-Length; returns None if unavailable."""
    try:
        resp = session.head(url, timeout=CONNECT_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length else None
    except requests.RequestException as exc:
        log.warning("HEAD request failed (%s); cannot determine remote size.", exc)
        return None


def server_supports_range(url: str, session: requests.Session) -> bool:
    try:
        resp = session.head(url, timeout=CONNECT_TIMEOUT, allow_redirects=True)
        return resp.headers.get("Accept-Ranges", "none").lower() == "bytes"
    except requests.RequestException:
        return False


def md5_of_file(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_vcf(url: str, dest: Path, session: requests.Session) -> None:
    """
    Download URL to dest with resume support and a tqdm progress bar.
    Retries up to MAX_RETRIES times with exponential back-off on transient errors.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote_size = remote_file_size(url, session)
    local_size = dest.stat().st_size if dest.exists() else 0

    if remote_size is not None and local_size == remote_size:
        log.info("File already fully downloaded (%s bytes). Skipping download.", remote_size)
        return

    can_resume = local_size > 0 and server_supports_range(url, session)
    if can_resume:
        log.info(
            "Resuming download from byte %s (remote size: %s).",
            local_size,
            remote_size or "unknown",
        )
    elif local_size > 0:
        log.warning(
            "Server does not support Range requests; restarting download from zero."
        )
        local_size = 0

    headers = {"Range": f"bytes={local_size}-"} if can_resume else {}
    mode = "ab" if can_resume else "wb"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(
                url,
                headers=headers,
                stream=True,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            ) as resp:
                if resp.status_code == 416:
                    # Range not satisfiable — file on disk is larger than remote
                    log.error(
                        "HTTP 416: local file (%s bytes) is larger than remote. "
                        "Delete %s and retry.",
                        local_size,
                        dest,
                    )
                    sys.exit(1)

                resp.raise_for_status()

                total = (
                    int(resp.headers.get("Content-Length", 0)) + local_size
                    if remote_size is None
                    else remote_size
                )

                bar = tqdm(
                    total=total or None,
                    initial=local_size,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=dest.name,
                    dynamic_ncols=True,
                )
                with open(dest, mode) as fh:
                    try:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            fh.write(chunk)
                            bar.update(len(chunk))
                    finally:
                        bar.close()

            log.info("Download complete: %s", dest)
            return

        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status in (400, 401, 403, 404):
                log.error("HTTP %s — not retrying: %s", status, exc)
                sys.exit(1)
            log.warning("HTTP error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)

        except (requests.ConnectionError, requests.Timeout) as exc:
            log.warning("Network error on attempt %d/%d: %s", attempt, MAX_RETRIES, exc)

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_BASE ** attempt
            log.info("Retrying in %s seconds...", wait)
            time.sleep(wait)
            # Update resume offset in case partial data was written
            local_size = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={local_size}-"} if can_resume else {}
            mode = "ab" if can_resume else "wb"

    log.error("Download failed after %d attempts.", MAX_RETRIES)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Validation and summary
# ---------------------------------------------------------------------------

def validate_and_summarise(vcf_path: Path) -> dict:
    """
    Open the VCF with cyvcf2, verify it is readable, and collect:
      - first N header lines
      - total variant count
      - per-chromosome variant counts
      - whether AF (allele frequency) INFO annotations are present
    Returns a summary dict.
    """
    try:
        from cyvcf2 import VCF
    except ImportError:
        log.error("cyvcf2 is not installed. Run: pip install cyvcf2")
        sys.exit(1)

    log.info("Opening VCF for validation: %s", vcf_path)

    try:
        vcf = VCF(str(vcf_path))
    except Exception as exc:
        log.error("cyvcf2 could not open the file: %s", exc)
        sys.exit(1)

    # -- Header check -------------------------------------------------------
    raw_header = vcf.raw_header.splitlines()
    first_lines = raw_header[:HEADER_LINES_TO_VALIDATE]
    if not first_lines or not first_lines[0].startswith("##fileformat=VCF"):
        log.error(
            "File does not appear to be a valid VCF. First header line: %r",
            first_lines[0] if first_lines else "<empty>",
        )
        sys.exit(1)

    log.info("VCF header OK. First %d lines:", HEADER_LINES_TO_VALIDATE)
    for line in first_lines:
        log.info("  %s", line)

    # -- AF INFO field detection --------------------------------------------
    info_ids = {h["ID"] for h in vcf.header_iter() if h.get("HeaderType") == "INFO"}
    has_af = any(k in info_ids for k in ("AF", "AF_SAS", "AF_INDIGEN", "IndiAF"))
    log.info(
        "INFO fields present: %s",
        ", ".join(sorted(info_ids)) if info_ids else "(none detected)",
    )
    log.info("Allele frequency (AF) annotations present: %s", has_af)

    # -- Variant enumeration ------------------------------------------------
    log.info(
        "Counting variants (capped at %s for large files)...",
        f"{VARIANT_COUNT_LIMIT:,}",
    )
    chrom_counts: Counter = Counter()
    total = 0

    for variant in vcf:
        chrom_counts[variant.CHROM] += 1
        total += 1
        if total % 500_000 == 0:
            log.info("  ... %s variants counted so far", f"{total:,}")
        if total >= VARIANT_COUNT_LIMIT:
            log.warning(
                "Reached variant count cap (%s). Counts are partial.",
                f"{VARIANT_COUNT_LIMIT:,}",
            )
            break

    vcf.close()

    log.info("Total variants counted: %s%s", f"{total:,}", "+" if total >= VARIANT_COUNT_LIMIT else "")
    log.info("Chromosome distribution:")
    for chrom, count in sorted(chrom_counts.items(), key=lambda x: x[1], reverse=True):
        log.info("  %-6s  %s", chrom, f"{count:,}")

    return {
        "vcf_path": str(vcf_path),
        "file_size_bytes": vcf_path.stat().st_size,
        "md5": md5_of_file(vcf_path),
        "header_first_lines": first_lines,
        "info_fields": sorted(info_ids),
        "has_af_annotation": has_af,
        "variant_count": total,
        "variant_count_capped": total >= VARIANT_COUNT_LIMIT,
        "chromosome_distribution": dict(chrom_counts),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and validate the IndiGenomes VCF."
    )
    parser.add_argument(
        "--url",
        required=True,
        help=(
            "Direct URL to the IndiGenomes .vcf.gz file. "
            "Obtain this from IGIB after your data-access request is approved."
        ),
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml in project root).",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download and run validation only (file must already exist).",
    )
    args = parser.parse_args()

    # Resolve config relative to project root (one level up from scripts/)
    root = Path(__file__).resolve().parents[1]
    config_path = root / args.config
    if not config_path.exists():
        log.error("Config file not found: %s", config_path)
        sys.exit(1)

    config = load_config(config_path)
    vcf_path, summary_path = resolve_paths(config)

    session = requests.Session()
    session.headers.update({"User-Agent": "dna-facial-approx/1.0 (academic research)"})

    # -- Download -----------------------------------------------------------
    if not args.skip_download:
        log.info("Destination: %s", vcf_path)
        download_vcf(args.url, vcf_path, session)
    else:
        if not vcf_path.exists():
            log.error("--skip-download set but file not found: %s", vcf_path)
            sys.exit(1)
        log.info("Skipping download; using existing file: %s", vcf_path)

    # -- Validate and summarise ---------------------------------------------
    summary = validate_and_summarise(vcf_path)

    # -- Write JSON summary -------------------------------------------------
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("Summary written to: %s", summary_path)

    # -- Final report -------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  IndiGenomes VCF summary")
    print("=" * 60)
    print(f"  File          : {vcf_path}")
    print(f"  Size          : {summary['file_size_bytes'] / 1e9:.2f} GB")
    print(f"  MD5           : {summary['md5']}")
    print(f"  Variants      : {summary['variant_count']:,}"
          + (" (partial count)" if summary["variant_count_capped"] else ""))
    print(f"  AF annotations: {'YES' if summary['has_af_annotation'] else 'NO -- check INFO fields'}")
    print(f"  Chromosomes   : {len(summary['chromosome_distribution'])}")
    print(f"  Summary JSON  : {summary_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

"""
download_gwas_sumstats.py

Downloads and indexes GWAS summary statistics for facial morphology.

Steps
-----
1. Download Xiong 2025 (DOI 10.5281/zenodo.13730680) via zenodo-get CLI,
   falling back to the Zenodo REST API.
2. Inspect SnpInfo.tsv — print columns, total SNP count, chromosome distribution.
3. Print top-10 genome-wide significant hits for 5 representative trait files.
4. Build data/processed/gwas_trait_index.csv (path, facial region, landmark pair,
   GWS SNP count at P < 5e-8).
5. For Du 2025: query CrossRef for the paper DOI, extract any data DOI from the
   metadata, and attempt a Zenodo/figshare download.

Usage
-----
    python scripts/download_gwas_sumstats.py [--config config.yaml]
    python scripts/download_gwas_sumstats.py --du2025-doi 10.XXXX/XXXXX
    python scripts/download_gwas_sumstats.py --skip-download
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import io
import json
import logging
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
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
XIONG2025_DOI   = "10.5281/zenodo.13730680"
ZENODO_API      = "https://zenodo.org/api/records"
CROSSREF_API    = "https://api.crossref.org/works"
GWS_P              = 5e-8
TOP_N_HITS         = 10
INSPECT_N_FILES    = 5
MAX_RETRIES        = 5
GWS_SCAN_LIMIT_MB  = 300   # skip full GWS scan for archives larger than this
BACKOFF         = 2       # seconds; doubles each retry
CONNECT_TIMEOUT = 30
READ_TIMEOUT    = 300

# Column-name aliases for heterogeneous GWAS file formats
COL_ALIASES: dict[str, set[str]] = {
    "chr":  {"CHR", "CHROM", "#CHROM", "Chromosome", "chr", "chrom"},
    "bp":   {"BP", "POS", "POSITION", "Position", "pos", "base_pair_location"},
    "snp":  {"SNP", "rsID", "RS_ID", "ID", "MarkerName", "variant_id"},
    "p":    {"P", "P_VALUE", "PVALUE", "p_value", "P.VALUE", "pval",
             "p_value_association"},
    "beta": {"BETA", "OR", "EFFECT", "Beta", "beta", "Effect", "effect_size"},
}

REGION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"nose|nasal",        re.I), "nose"),
    (re.compile(r"eye|orbit|ocular",  re.I), "eye"),
    (re.compile(r"mouth|lip|oral",    re.I), "mouth"),
    (re.compile(r"ear",               re.I), "ear"),
    (re.compile(r"jaw|mandib|chin",   re.I), "jaw"),
    (re.compile(r"cheek|zygom",       re.I), "cheek"),
    (re.compile(r"forehead|frontal",  re.I), "forehead"),
    (re.compile(r"face|facial|skull", re.I), "face_general"),
]
LANDMARK_RE = re.compile(
    r"[Ll][Mm]?\d+[-_][Ll][Mm]?\d+|[Ll]\d+[Rr]\d+|\d{2,3}[-_]\d{2,3}"
)

# Stems / suffixes that are NOT trait summary stat files
SKIP_STEMS    = {"readme", "snpinfo", "snp_info", "manifest", "index", "md5"}
SKIP_SUFFIXES = {".json", ".md", ".pdf", ".png", ".R", ".py", ".sh", ".log"}


# ===========================================================================
# Config
# ===========================================================================

def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def gwas_dirs(config: dict, root: Path) -> tuple[Path, Path, Path]:
    """Return (gwas_raw_dir, xiong_dir, processed_dir)."""
    raw  = root / config["paths"]["gwas_sumstats_dir"]
    proc = root / config["paths"]["processed_dir"]
    return raw, raw / "xiong2025", proc


# ===========================================================================
# Step 1 — Download Xiong 2025
# ===========================================================================

def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "dna-facial-approx/1.0 (academic research)"
    return s


def _retry_get(session: requests.Session, url: str, **kwargs) -> requests.Response:
    """GET with exponential-backoff retry (skips 4xx immediately)."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=(CONNECT_TIMEOUT, READ_TIMEOUT), **kwargs)
            if resp.status_code in (400, 401, 403, 404):
                resp.raise_for_status()
            resp.raise_for_status()
            return resp
        except requests.HTTPError:
            raise
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF ** attempt)
    raise RuntimeError(f"All {MAX_RETRIES} attempts failed for {url}")


def _stream_download(session: requests.Session, url: str, dest: Path) -> None:
    """Stream url → dest with resume support."""
    local = dest.stat().st_size if dest.exists() else 0

    # Skip if already complete
    try:
        head = session.head(url, timeout=CONNECT_TIMEOUT, allow_redirects=True)
        remote = int(head.headers.get("Content-Length", 0))
        if remote and local == remote:
            log.info("Already complete, skipping: %s", dest.name)
            return
    except requests.RequestException:
        pass

    headers = {"Range": f"bytes={local}-"} if local else {}
    mode    = "ab" if local else "wb"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, headers=headers, stream=True,
                             timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)) as resp:
                if resp.status_code == 416:          # range not satisfiable → restart
                    local, headers, mode = 0, {}, "wb"
                    continue
                resp.raise_for_status()
                with open(dest, mode) as fh:
                    for chunk in resp.iter_content(1024 * 1024):
                        fh.write(chunk)
            log.info("Downloaded: %s", dest.name)
            return
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "?"
            if status in (400, 401, 403, 404):
                log.error("HTTP %s — not retrying: %s", status, dest.name)
                return
            log.warning("Attempt %d/%d HTTP error: %s", attempt, MAX_RETRIES, exc)
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d network error: %s", attempt, MAX_RETRIES, exc)
        if attempt < MAX_RETRIES:
            time.sleep(BACKOFF ** attempt)
            local = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={local}-"} if local else {}
            mode    = "ab" if local else "wb"

    log.error("Failed to download %s after %d attempts.", dest.name, MAX_RETRIES)


def download_zenodo(doi: str, out_dir: Path, session: requests.Session) -> None:
    """Download all files from a Zenodo DOI.  Tries zenodo-get CLI first."""
    out_dir.mkdir(parents=True, exist_ok=True)

    if shutil.which("zenodo_get"):
        log.info("Using zenodo_get CLI for %s", doi)
        r = subprocess.run(["zenodo_get", doi, "-o", str(out_dir)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            log.info("zenodo_get finished successfully.")
            return
        log.warning("zenodo_get exited %d; falling back to REST API.\n%s",
                    r.returncode, r.stderr.strip())
    else:
        log.info("zenodo_get not found; using Zenodo REST API for %s", doi)

    record_id = doi.rstrip("/").split(".")[-1]
    url  = f"{ZENODO_API}/{record_id}"
    data = _retry_get(session, url).json()
    files = data.get("files", [])
    if not files:
        log.error("No files in Zenodo record %s.", record_id)
        sys.exit(1)

    log.info("Record %s has %d file(s).", record_id, len(files))
    for f in files:
        fname    = f.get("key") or f.get("filename") or "unknown"
        links    = f.get("links", {})
        file_url = links.get("self") or links.get("download")
        if not file_url:
            log.warning("No URL for %s — skipping.", fname)
            continue
        _stream_download(session, file_url, out_dir / fname)


# ===========================================================================
# Step 2 — Inspect SnpInfo.tsv
# ===========================================================================

class _TarTsvContext:
    """Context manager that concatenates all *.tsv members of a tar.bz2 archive."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._tf: Optional[tarfile.TarFile] = None
        self._buf: Optional[io.StringIO] = None

    def __enter__(self) -> io.StringIO:
        self._tf = tarfile.open(self._path, "r:bz2")
        members = [m for m in self._tf.getmembers() if m.isfile() and m.name.endswith(".tsv")]
        if not members:
            raise ValueError(f"No .tsv files found inside {self._path.name}")

        parts: list[str] = []
        for i, member in enumerate(members):
            fh = self._tf.extractfile(member)
            if fh is None:
                continue
            text = fh.read().decode("utf-8", errors="replace")
            if i == 0:
                parts.append(text)
            else:
                # Skip repeated header line for subsequent members
                _, _, body = text.partition("\n")
                parts.append(body)

        self._buf = io.StringIO("".join(parts))
        return self._buf

    def __exit__(self, *_: object) -> None:
        if self._tf is not None:
            self._tf.close()


def _open_tsv(path: Path):
    """Open a TSV file regardless of compression.  Handles .gz/.bgz, .bz2, .tar.bz2."""
    name = path.name.lower()
    if name.endswith(".tar.bz2") or name.endswith(".tar.gz"):
        return _TarTsvContext(path)
    if path.suffix.lower() in (".bz2", ".bz"):
        return bz2.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".gz", ".bgz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _detect_cols(header: list[str]) -> dict[str, Optional[str]]:
    """Map logical role → actual column name found in header."""
    return {
        role: next((h for h in header if h in aliases), None)
        for role, aliases in COL_ALIASES.items()
    }


def inspect_snpinfo(path: Path) -> None:
    chrom_counts: Counter = Counter()
    total = 0
    cols: list[str] = []

    with _open_tsv(path) as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        cols   = list(reader.fieldnames or [])
        chr_col = _detect_cols(cols).get("chr")
        for row in reader:
            total += 1
            if chr_col and chr_col in row:
                chrom_counts[row[chr_col]] += 1

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  SnpInfo: {path.name}")
    print(bar)
    print(f"  Columns ({len(cols)}): {', '.join(cols)}")
    print(f"  Total SNPs : {total:,}")
    if chrom_counts:
        print(f"  Chromosomes ({len(chrom_counts)}):")
        for chrom in sorted(chrom_counts, key=lambda c: (len(c), c)):
            print(f"    chr{chrom:<4}  {chrom_counts[chrom]:>10,}")
    print(f"{bar}\n")


# ===========================================================================
# Step 3 — Top GWS hits
# ===========================================================================

def _is_sumstat(path: Path) -> bool:
    return (
        not any(s in path.stem.lower() for s in SKIP_STEMS)
        and path.suffix.lower() not in SKIP_SUFFIXES
    )


def _peek_columns(path: Path, nrows: int = 3) -> list[str]:
    """Read only the header of the first TSV member — fast for any file size."""
    name = path.name.lower()
    if name.endswith(".tar.bz2"):
        # Use r:bz2 (seekable bz2.BZ2File) with tf.next() — reads only the first
        # member's header chain (PAX or POSIX) and 8 KB of data, not the whole archive.
        # getmembers() decompresses everything; tf.next() stops after the first member.
        try:
            with tarfile.open(path, "r:bz2") as tf:
                member = tf.next()
                while member is not None:
                    if member.isfile() and member.name.endswith(".tsv"):
                        fh = tf.extractfile(member)
                        if fh is not None:
                            cols = list(pd.read_csv(io.BytesIO(fh.read(8192)), sep="\t",
                                                    nrows=nrows,
                                                    encoding_errors="replace").columns)
                            if cols:
                                return cols
                        break
                    member = tf.next()
        except Exception:
            pass
        return []
    with _open_tsv(path) as fh:
        return list(pd.read_csv(fh, sep="\t", nrows=nrows,
                                encoding_errors="replace").columns)


def _scan_gws_streaming(
    path: Path,
    p_col: str,
    top_n: int = TOP_N_HITS,
) -> tuple[list[dict], int, int]:
    """Stream only the p-value column in 200k-row chunks via a single archive pass.

    Returns (top_n_hits_sorted, gws_count, total_rows).
    Returns ([], -1, -1) when the archive exceeds GWS_SCAN_LIMIT_MB.
    """
    size_mb = path.stat().st_size / 1_048_576
    if size_mb > GWS_SCAN_LIMIT_MB:
        log.info(
            "GWS scan skipped for large archive (%.0f MB > %d MB limit): %s",
            size_mb, GWS_SCAN_LIMIT_MB, path.name,
        )
        return [], -1, -1

    gws_rows: list[dict] = []
    gws_n = 0
    total = 0

    def _scan(src, compression="infer") -> None:
        nonlocal gws_n, total
        for chunk in pd.read_csv(src, sep="\t", usecols=[p_col],
                                  chunksize=200_000, encoding_errors="replace",
                                  compression=compression):
            p = pd.to_numeric(chunk[p_col], errors="coerce")
            mask = p < GWS_P
            gws_n += int(mask.sum())
            total += len(chunk)
            if mask.any():
                gws_rows.extend(chunk[mask].assign(_p=p[mask]).to_dict("records"))

    name = path.name.lower()
    if name.endswith(".tar.bz2"):
        # Extract to a temp dir (r:bz2 non-streaming) then scan plain files.
        # r|bz2 streaming mode causes EOFError when fh.read() is called after
        # the iterator has advanced past the member.
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with tarfile.open(path, "r:bz2") as tf:
                tsv_members = [m for m in tf.getmembers()
                               if m.isfile() and m.name.endswith(".tsv")]
                tf.extractall(tmpdir, members=tsv_members)
            for member in tsv_members:
                extracted = tmp_path / member.name
                if extracted.exists():
                    _scan(extracted, compression=None)
    elif path.suffix.lower() in (".bz2", ".bz"):
        _scan(path, compression="bz2")
    elif path.suffix.lower() in (".gz", ".bgz"):
        _scan(path, compression="gzip")
    else:
        _scan(path, compression=None)

    gws_rows.sort(key=lambda r: r["_p"])
    return gws_rows[:top_n], gws_n, total


def _top_hits(path: Path, n: int = TOP_N_HITS
              ) -> tuple[list[dict], dict[str, Optional[str]], int]:
    """Return (top-n rows sorted by p, col_map, gws_count)."""
    cols    = _peek_columns(path)
    col_map = _detect_cols(cols)
    p_col   = col_map["p"]
    if not p_col:
        log.warning("%s: no p-value column detected. Columns: %s", path.name, cols)
        return [], col_map, 0

    hits, gws_n, _ = _scan_gws_streaming(path, p_col, top_n=n)
    if gws_n == -1:
        gws_n = 0
    return hits, col_map, gws_n


def print_top_hits(path: Path) -> int:
    """Print top GWS hits table for one trait file; returns GWS count."""
    hits, col_map, gws_n = _top_hits(path)
    snp_col  = col_map.get("snp")
    chr_col  = col_map.get("chr")
    bp_col   = col_map.get("bp")
    p_col    = col_map.get("p")
    beta_col = col_map.get("beta")

    print(f"\n  Trait : {path.name}")
    print(f"  GWS hits (P < {GWS_P:.0e}): {gws_n:,}")
    if not hits:
        print("  (no genome-wide significant hits)\n")
        return 0

    hdr = f"  {'SNP':<18} {'CHR':<5} {'BP':<12} {'P':>12} {'BETA':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for row in hits:
        print(
            f"  {str(row.get(snp_col, 'NA') if snp_col else 'NA'):<18}"
            f" {str(row.get(chr_col,  'NA') if chr_col  else 'NA'):<5}"
            f" {str(row.get(bp_col,   'NA') if bp_col   else 'NA'):<12}"
            f" {row['_p']:>12.3e}"
            f" {str(row.get(beta_col, 'NA') if beta_col else 'NA'):>10}"
        )
    return gws_n


# ===========================================================================
# Step 4 — Trait index
# ===========================================================================

def _infer_region(stem: str) -> str:
    for pat, label in REGION_PATTERNS:
        if pat.search(stem):
            return label
    return "unknown"


def _infer_landmark(stem: str) -> str:
    m = LANDMARK_RE.search(stem)
    return m.group(0) if m else "unknown"


def _count_snps(path: Path) -> int:
    """Count total rows; returns -1 for archives above GWS_SCAN_LIMIT_MB."""
    cols    = _peek_columns(path)
    col_map = _detect_cols(cols)
    p_col   = col_map.get("p") or (cols[0] if cols else None)
    if not p_col:
        return -1
    _, _, total = _scan_gws_streaming(path, p_col, top_n=0)
    return total


def build_trait_index(gwas_raw: Path, out_csv: Path) -> int:
    trait_files = sorted(
        p for p in gwas_raw.rglob("*") if p.is_file() and _is_sumstat(p)
    )
    if not trait_files:
        log.warning("No trait files found under %s.", gwas_raw)
        return 0

    log.info("Indexing %d trait files …", len(trait_files))
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "trait_file", "dataset", "facial_region", "landmark_pair",
            "n_gws_hits", "n_total_snps", "p_col", "snp_col", "top_snp", "top_p",
        ])
        for path in trait_files:
            try:
                hits, col_map, gws_n = _top_hits(path, n=1)
                total = _count_snps(path)
                p_col   = col_map.get("p") or ""
                snp_col = col_map.get("snp") or ""
                top_snp = hits[0].get(snp_col, "") if hits and snp_col else ""
                top_p   = f"{hits[0]['_p']:.3e}" if hits else ""
            except Exception as exc:
                log.warning("Skipping %s: %s", path.name, exc)
                p_col = snp_col = top_snp = top_p = ""
                gws_n = total = 0

            writer.writerow([
                str(path), path.parent.name,
                _infer_region(path.stem), _infer_landmark(path.stem),
                gws_n, total, p_col, snp_col, top_snp, top_p,
            ])

    log.info("Trait index written → %s (%d rows)", out_csv, len(trait_files))
    return len(trait_files)


def print_index_preview(out_csv: Path, n: int = 15) -> None:
    with open(out_csv) as fh:
        rows = list(csv.DictReader(fh))
    print(f"\n  {'File':<38} {'Region':<14} {'Landmarks':<18} {'GWS':>7} {'Total':>9}")
    print("  " + "-" * 90)
    for row in rows[:n]:
        print(
            f"  {Path(row['trait_file']).name[:36]:<38}"
            f" {row['facial_region'][:12]:<14}"
            f" {row['landmark_pair'][:16]:<18}"
            f" {row['n_gws_hits']:>7}"
            f" {row['n_total_snps']:>9}"
        )
    if len(rows) > n:
        print(f"  … and {len(rows) - n} more rows.")
    print()


# ===========================================================================
# Step 5 — Du 2025 data availability
# ===========================================================================

def _crossref_message(doi: str, session: requests.Session) -> Optional[dict]:
    headers = {"User-Agent": "dna-facial-approx/1.0 (mailto:archisha.nambiar@gmail.com)"}
    try:
        resp = _retry_get(session, f"{CROSSREF_API}/{doi}", headers=headers)
        return resp.json().get("message", {})
    except Exception as exc:
        log.warning("CrossRef query failed: %s", exc)
        return None


def _extract_data_dois(msg: dict) -> list[str]:
    """Pull Zenodo/figshare/OSF/Dryad DOIs from CrossRef metadata."""
    found: list[str] = []
    data_re = re.compile(r"zenodo|figshare|osf\.io|dryad", re.I)

    for rel_list in msg.get("relation", {}).values():
        for item in rel_list:
            if isinstance(item, dict):
                val = item.get("id", "")
                if data_re.search(val):
                    found.append(val)

    for link in msg.get("link", []):
        url = link.get("URL", "")
        m   = re.search(r"10\.\d{4,9}/\S+", url)
        if m and data_re.search(url):
            found.append(m.group(0))

    for m in re.finditer(
        r"10\.5281/zenodo\.\d+|10\.\d{4}/figshare\.\d+",
        msg.get("abstract", ""),
    ):
        found.append(m.group(0))

    return list(dict.fromkeys(found))


def check_du2025(doi: str, du_dir: Path, session: requests.Session) -> None:
    log.info("Querying CrossRef for Du 2025 (DOI: %s) …", doi)
    msg = _crossref_message(doi, session)

    bar = "=" * 62
    print(f"\n{bar}")
    print("  Du 2025 — data availability")
    print(bar)

    if not msg:
        print("  CrossRef returned no data — DOI may be wrong or not yet indexed.")
        print(f"  Try visiting: https://doi.org/{doi}")
        print(f"{bar}\n")
        return

    titles  = msg.get("title", [])
    authors = msg.get("author", [])
    year    = (msg.get("published", {}).get("date-parts") or [[""]])[0][0]
    print(f"  DOI    : {doi}")
    print(f"  Title  : {titles[0] if titles else 'N/A'}")
    print(f"  Author : {authors[0].get('family', '?') if authors else '?'} et al. ({year})")

    data_dois = _extract_data_dois(msg)
    if data_dois:
        print(f"  Data DOI(s): {data_dois}")
        for ddoi in data_dois:
            if "zenodo" in ddoi.lower():
                log.info("Downloading data DOI %s → %s", ddoi, du_dir)
                download_zenodo(ddoi, du_dir, session)
            else:
                print(f"  Non-Zenodo repository: {ddoi}")
                print(f"  -> Download manually and place under {du_dir}")
    else:
        print("  No machine-readable data DOI found in CrossRef metadata.")
        print("  Action: visit the paper and check the Data Availability section.")
        print(f"  Paper URL : https://doi.org/{doi}")
        print(f"  Place files under: {du_dir}")

    print(f"{bar}\n")


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    # Ensure stdout handles non-ASCII characters on Windows (cp1252 terminals)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Download and index GWAS summary statistics for facial morphology."
    )
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--du2025-doi",   default=None, metavar="DOI",
                        help="CrossRef DOI for Du 2025 paper")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip downloads; inspect and index existing files only")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg  = load_config(root / args.config)
    gwas_raw, xiong_dir, processed_dir = gwas_dirs(cfg, root)
    trait_index = processed_dir / "gwas_trait_index.csv"
    du_dir      = gwas_raw / "du2025"

    session = _session()

    # ------------------------------------------------------------------
    # Step 1: Download Xiong 2025
    # ------------------------------------------------------------------
    log.info("=== Step 1: Download Xiong 2025 ===")
    if args.skip_download:
        log.info("--skip-download set; skipping.")
    else:
        download_zenodo(XIONG2025_DOI, xiong_dir, session)

    # ------------------------------------------------------------------
    # Step 2: Inspect SnpInfo.tsv
    # ------------------------------------------------------------------
    log.info("=== Step 2: Inspect SnpInfo.tsv ===")
    snpinfo_files = (
        list(xiong_dir.glob("*[Ss]np[Ii]nfo*")) +
        list(xiong_dir.glob("*SNP_INFO*"))
    )
    if snpinfo_files:
        inspect_snpinfo(snpinfo_files[0])
    else:
        log.warning("SnpInfo file not found under %s. Run without --skip-download first.",
                    xiong_dir)

    # ------------------------------------------------------------------
    # Step 3: Top GWS hits for 5 representative trait files
    # ------------------------------------------------------------------
    log.info("=== Step 3: Top %d GWS hits for %d representative trait files ===",
             TOP_N_HITS, INSPECT_N_FILES)
    trait_files = sorted(
        p for p in xiong_dir.rglob("*") if p.is_file() and _is_sumstat(p)
    )
    if trait_files:
        bar = "=" * 62
        print(f"\n{bar}")
        print(f"  Top {TOP_N_HITS} GWS hits  (P < {GWS_P:.0e})  — Xiong 2025")
        print(bar)
        for tf in trait_files[:INSPECT_N_FILES]:
            print_top_hits(tf)
        print(f"{bar}\n")
    else:
        log.warning("No trait files found under %s.", xiong_dir)

    # ------------------------------------------------------------------
    # Step 4: Build trait index
    # ------------------------------------------------------------------
    log.info("=== Step 4: Build trait index ===")
    n = build_trait_index(gwas_raw, trait_index)
    if n:
        print_index_preview(trait_index)

    # ------------------------------------------------------------------
    # Step 5: Du 2025 data availability
    # ------------------------------------------------------------------
    log.info("=== Step 5: Du 2025 data availability ===")
    if args.du2025_doi:
        check_du2025(args.du2025_doi, du_dir, session)
    else:
        log.info("No --du2025-doi supplied.")
        log.info("Run:  python scripts/download_gwas_sumstats.py --du2025-doi 10.XXXX/XXXXX")


if __name__ == "__main__":
    main()

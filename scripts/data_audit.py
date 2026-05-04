"""
data_audit.py

Produces a comprehensive audit report saved as docs/data_audit.md.

Sections
--------
1. Dataset inventory  — size, variant count, sample count, genome build
2. HIrisPlex-S SNP coverage  — 41 SNPs vs 1000G SAS VCF
3. GWAS morphology SNP coverage  — GWS + LD-clumped SNPs vs 1000G SAS VCF
4. Genome build consistency  — mismatch warnings
5. Pipeline readiness checklist  — green/red per stage

Notes
-----
VCF inspection requires cyvcf2 (pip install cyvcf2).
Large VCFs (1KG, IndiGenomes) are iterated in a single pass —
expect several minutes per file if they are multi-GB.

Usage
-----
    python scripts/data_audit.py [--config config.yaml]
    python scripts/data_audit.py --skip-vcf-scan   # skip variant counting (fast mode)
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GWS_P       = 5e-8
LD_CLUMP_KB = 500   # physical clumping window (proxy for r²)

BUILD_RE = re.compile(
    r"(GRCh37|hg19|b37|hs37|NCBI36|GRCh38|hg38|b38)",
    re.I,
)

COL_ALIASES: dict[str, set[str]] = {
    "chr": {"CHR", "CHROM", "#CHROM", "Chromosome", "chr", "chrom"},
    "bp":  {"BP", "POS", "POSITION", "Position", "pos", "base_pair_location"},
    "snp": {"SNP", "rsID", "RS_ID", "ID", "MarkerName", "variant_id"},
    "p":   {"P", "P_VALUE", "PVALUE", "p_value", "P.VALUE", "pval",
            "p_value_association"},
}

SKIP_STEMS    = {"readme", "snpinfo", "snp_info", "manifest", "index", "md5"}
SKIP_SUFFIXES = {".json", ".md", ".pdf", ".png", ".R", ".py", ".sh", ".log", ".csv"}


# ===========================================================================
# VCF inspection
# ===========================================================================

def _detect_build(header_text: str) -> str:
    m = BUILD_RE.search(header_text)
    if m:
        token = m.group(0).lower()
        if token in ("grch37", "hg19", "b37", "hs37", "ncbi36"):
            return "GRCh37/hg19"
        if token in ("grch38", "hg38", "b38"):
            return "GRCh38/hg38"

    # Heuristic: ##contig IDs with/without "chr" prefix
    contig_ids = re.findall(r"##contig=<ID=([^,>]+)", header_text)
    if contig_ids:
        has_chr = any(c.startswith("chr") for c in contig_ids)
        return "GRCh38/hg38 (chr-prefix inferred)" if has_chr else "GRCh37/hg19 (no-chr-prefix inferred)"
    return "unknown"


def inspect_vcf(path: Path, collect_ids: bool = False,
                skip_scan: bool = False) -> tuple[dict, set[str]]:
    """
    Single-pass VCF inspection.

    Returns (info_dict, rsid_set).
    info_dict keys: size_bytes, n_variants, n_samples, build, error
    rsid_set: populated only when collect_ids=True and skip_scan=False
    """
    info: dict = {
        "size_bytes": path.stat().st_size if path.exists() else None,
        "n_variants": None,
        "n_samples":  None,
        "build":      "unknown",
        "error":      None,
    }
    rsids: set[str] = set()

    if not path.exists():
        info["error"] = "file not found"
        return info, rsids

    if skip_scan:
        info["n_variants"] = "(skipped)"
        info["n_samples"]  = "(skipped)"
        return info, rsids

    try:
        import cyvcf2  # noqa: PLC0415
    except ImportError:
        info["error"] = "cyvcf2 not installed"
        return info, rsids

    try:
        vcf = cyvcf2.VCF(str(path))
        info["build"]     = _detect_build(vcf.raw_header)
        info["n_samples"] = len(vcf.samples)
        n = 0
        for var in vcf:
            n += 1
            if collect_ids and var.ID and var.ID != ".":
                rsids.add(var.ID)
        info["n_variants"] = n
        vcf.close()
    except Exception as exc:
        info["error"] = str(exc)

    return info, rsids


# ===========================================================================
# HIrisPlex helpers
# ===========================================================================

def read_hirisplex_snps(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with open(csv_path) as fh:
        return list(csv.DictReader(fh))


# ===========================================================================
# GWAS helpers
# ===========================================================================

def _detect_cols(header: list[str]) -> dict[str, Optional[str]]:
    return {
        role: next((h for h in header if h in aliases), None)
        for role, aliases in COL_ALIASES.items()
    }


def _open_tsv(path: Path):
    if path.suffix.lower() in (".gz", ".bgz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _is_sumstat(path: Path) -> bool:
    return (
        not any(s in path.stem.lower() for s in SKIP_STEMS)
        and path.suffix.lower() not in SKIP_SUFFIXES
    )


def _physical_clump(hits: list[dict]) -> list[dict]:
    """Greedy ±LD_CLUMP_KB distance clumping; best-p hit wins each window."""
    by_p = sorted(hits, key=lambda r: r["p"])
    kept: list[dict] = []
    for hit in by_p:
        chrom = hit["chr"]
        bp    = hit["bp"]
        if bp < 0:
            kept.append(hit)
            continue
        if not any(k["chr"] == chrom and abs(k["bp"] - bp) < LD_CLUMP_KB * 1000
                   for k in kept):
            kept.append(hit)
    return kept


def load_gwas_gws_hits(gwas_dir: Path) -> dict[str, list[dict]]:
    """Return {trait_stem: [gws_hit, ...]} for every sumstat file found."""
    results: dict[str, list[dict]] = {}
    if not gwas_dir.exists():
        return results

    for path in sorted(p for p in gwas_dir.rglob("*")
                       if p.is_file() and _is_sumstat(p)):
        hits: list[dict] = []
        try:
            with _open_tsv(path) as fh:
                reader = csv.DictReader(fh, delimiter="\t")
                if not reader.fieldnames:
                    continue
                col = _detect_cols(list(reader.fieldnames))
                if not col["p"]:
                    continue
                for row in reader:
                    try:
                        pval = float(row[col["p"]])
                    except (ValueError, TypeError):
                        continue
                    if pval >= GWS_P:
                        continue
                    hits.append({
                        "snp": row.get(col["snp"], "") if col["snp"] else "",
                        "chr": row.get(col["chr"], "") if col["chr"] else "",
                        "bp":  _safe_int(row.get(col["bp"], "")) if col["bp"] else -1,
                        "p":   pval,
                    })
        except Exception as exc:
            log.warning("Could not parse %s: %s", path.name, exc)
            continue
        results[path.stem] = hits

    return results


def _safe_int(val: str) -> int:
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return -1


# ===========================================================================
# Markdown formatting helpers
# ===========================================================================

def _fmt_size(n: Optional[int]) -> str:
    if n is None:
        return "N/A"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _ok(cond: bool) -> str:
    return "✅" if cond else "❌"


class MD:
    """Tiny markdown builder."""

    def __init__(self) -> None:
        self._lines: list[str] = []

    def raw(self, text: str) -> None:
        self._lines.append(text)

    def h(self, text: str, level: int = 2) -> None:
        self._lines.append(f"\n{'#' * level} {text}\n")

    def p(self, text: str) -> None:
        self._lines.append(text + "\n")

    def warn(self, text: str) -> None:
        self._lines.append(f"> ⚠️ **{text}**\n")

    def table(self, headers: list[str], rows: list[list]) -> None:
        self._lines.append("| " + " | ".join(headers) + " |")
        self._lines.append("| " + " | ".join("-" * max(len(h), 3) for h in headers) + " |")
        for row in rows:
            self._lines.append("| " + " | ".join(str(c) for c in row) + " |")
        self._lines.append("")

    def build(self) -> str:
        return "\n".join(self._lines)


# ===========================================================================
# Report sections
# ===========================================================================

def section_inventory(
    md: MD,
    datasets: dict[str, tuple[Path, dict]],
    hiri_snps: list[dict],
    hirisplex_csv_path: Path,
    gwas_dir: Path,
) -> None:
    md.h("1. Dataset Inventory")
    md.p("File size, variant count, sample count, and inferred genome build for each dataset.")

    rows: list[list] = []
    for label, (path, info) in datasets.items():
        status = f"*{info['error']}*" if info["error"] else "OK"
        rows.append([
            label,
            f"`{path.name}`",
            _fmt_size(info["size_bytes"]),
            _fmt(info["n_variants"]),
            _fmt(info["n_samples"]),
            info["build"],
            status,
        ])

    # HIrisPlex CSV row
    rows.append([
        "HIrisPlex-S SNP list",
        f"`{hirisplex_csv_path.name}`",
        _fmt_size(hirisplex_csv_path.stat().st_size if hirisplex_csv_path.exists() else None),
        f"{len(hiri_snps)} SNPs",
        "N/A",
        "N/A",
        "OK" if hirisplex_csv_path.exists() else "*file not found*",
    ])

    # GWAS directory row
    gwas_files = [f for f in gwas_dir.rglob("*") if f.is_file()] if gwas_dir.exists() else []
    gwas_bytes = sum(f.stat().st_size for f in gwas_files) if gwas_files else None
    rows.append([
        "GWAS sumstats",
        f"`{gwas_dir.name}/`",
        _fmt_size(gwas_bytes),
        f"{len(gwas_files)} files",
        "N/A",
        "N/A",
        "OK" if gwas_files else "*no files downloaded*",
    ])

    md.table(
        ["Dataset", "File", "Size", "Variants / SNPs", "Samples", "Build", "Status"],
        rows,
    )


def section_hirisplex(md: MD, hiri_snps: list[dict], kg_ids: set[str],
                      kg_missing: bool) -> None:
    md.h("2. HIrisPlex-S SNP Coverage (41 SNPs vs 1000G SAS VCF)")

    if kg_missing:
        md.warn("1000 Genomes VCF not found — HIrisPlex overlap analysis skipped.")
        return
    if not hiri_snps:
        md.warn("HIrisPlex-S SNP list not found — overlap analysis skipped.")
        return

    n_present = sum(1 for s in hiri_snps if s.get("rsID", "") in kg_ids)
    pct = n_present / len(hiri_snps) * 100

    md.p(f"**Coverage: {n_present}/{len(hiri_snps)} SNPs present in 1KG SAS VCF ({pct:.1f}%)**\n")

    missing_ids = [s["rsID"] for s in hiri_snps if s.get("rsID", "") not in kg_ids]
    if missing_ids:
        md.p(f"Missing SNPs: `{'`, `'.join(missing_ids)}`\n")
        md.p(
            "> These SNPs must be imputed or replaced with proxy variants "
            "before HIrisPlex-S scoring.\n"
        )

    rows = [
        [
            s.get("rsID", ""),
            s.get("gene", ""),
            s.get("chromosome", ""),
            s.get("trait_model", ""),
            _ok(s.get("rsID", "") in kg_ids),
        ]
        for s in hiri_snps
    ]
    md.table(["rsID", "Gene", "Chr", "Trait Model", "In 1KG SAS"], rows)


def section_gwas(md: MD, gwas_dir: Path, kg_ids: set[str], kg_missing: bool) -> None:
    md.h("3. GWAS Facial Morphology SNP Coverage")
    md.p(
        f"GWS threshold: P < {GWS_P:.0e}.  "
        f"LD filter: physical clumping ±{LD_CLUMP_KB} kb "
        f"(proxy for r² — no LD matrix available at audit time)."
    )

    gwas_files = [f for f in gwas_dir.rglob("*") if f.is_file() and _is_sumstat(f)] \
        if gwas_dir.exists() else []

    if not gwas_files:
        md.warn("No GWAS sumstat files found — run download_gwas_sumstats.py first.")
        return

    log.info("Loading GWS hits from %d GWAS files …", len(gwas_files))
    trait_hits = load_gwas_gws_hits(gwas_dir)

    if not trait_hits:
        md.warn("GWAS files present but no GWS hits extracted (check column names / format).")
        return

    rows: list[list] = []
    total_raw = total_clumped = total_in_kg = 0

    for trait, hits in sorted(trait_hits.items()):
        clumped = _physical_clump(hits)
        in_kg   = [h for h in clumped if h["snp"] and h["snp"] in kg_ids] \
                  if not kg_missing else []
        total_raw      += len(hits)
        total_clumped  += len(clumped)
        total_in_kg    += len(in_kg)
        cov = f"{len(in_kg)/len(clumped)*100:.0f}%" if clumped and not kg_missing else "N/A"
        rows.append([trait[:45], f"{len(hits):,}", f"{len(clumped):,}", f"{len(in_kg):,}", cov])

    kg_note = " (1KG VCF absent)" if kg_missing else ""
    md.p(
        f"**Totals across {len(trait_hits)} traits — "
        f"{total_raw:,} raw GWS → "
        f"{total_clumped:,} after LD clump → "
        f"{total_in_kg:,} in 1KG SAS{kg_note}**\n"
    )
    md.table(["Trait", "GWS (raw)", "After LD clump", "In 1KG SAS", "Coverage"], rows)


def section_build(md: MD, builds: dict[str, str]) -> None:
    md.h("4. Genome Build Consistency")

    present = {k: v for k, v in builds.items() if v not in ("unknown", "file not found")}
    unique  = set(present.values())

    if not present:
        md.p("No VCF files found — cannot assess build consistency.")
        return

    md.table(["Dataset", "Inferred Build"],
             [[label, build] for label, build in builds.items()])

    if len(unique) > 1:
        md.warn(
            "Build mismatch detected! Datasets appear to be on different genome builds. "
            "All VCFs must be on the same build before running the pipeline."
        )
        md.p(
            "**Action:** liftover mismatching datasets with `CrossMap` or "
            "`picard LiftoverVcf` before proceeding."
        )
    else:
        build = next(iter(unique)) if unique else "unknown"
        md.p(f"✅ All inspected VCFs report the same build: **{build}**")

    n_unknown = sum(1 for v in builds.values() if v == "unknown")
    if n_unknown:
        md.warn(
            f"{n_unknown} dataset(s) returned 'unknown' build "
            f"(file absent or no build tag in header) — verify manually."
        )


def section_checklist(
    md: MD,
    paths: dict[str, Path],
    gwas_file_count: int,
) -> None:
    md.h("5. Pipeline Readiness Checklist")

    def chk(path_or_cond, label: str, note: str) -> list:
        ok = path_or_cond if isinstance(path_or_cond, bool) else path_or_cond.exists()
        return [_ok(ok), label, note]

    p = paths
    items = [
        chk(p["input_vcf"],       "Input subject VCF",             "entry point for all stages"),
        chk(p["indigenomes"],      "IndiGenomes VCF",               "population allele-frequency reference"),
        chk(p["kg"],               "1000 Genomes SAS+EUR VCF",      "ancestry PCA reference panel"),
        chk(p["hiri_csv"],         "HIrisPlex-S SNP list CSV",      "41-SNP pigmentation manifest"),
        chk(p["hiri_coef"],        "HIrisPlex-S coefficients JSON", "model weights — fill TODOs from Walsh 2017"),
        chk(gwas_file_count > 0,   "GWAS sumstats (≥1 file)",       "Xiong 2025 / Du 2025"),
        chk(p["gwas_index"],       "GWAS trait index CSV",          "built by download_gwas_sumstats.py"),
        chk(p["face_templates"],   "Face template directory",       "base images + landmark annotations"),
        chk(p["output_dir"],       "outputs/ directory",            "writable output location"),
        chk(p["processed_dir"],    "data/processed/ directory",     "intermediate file store"),
    ]

    md.table(["", "Requirement", "Notes"], items)

    n_ok = sum(1 for r in items if r[0] == "✅")
    md.p(f"**{n_ok}/{len(items)} requirements met ({n_ok/len(items)*100:.0f}% ready)**")

    md.h("Stage-level Summary", level=3)

    stages = [
        ("Stage 1 · Preprocessing",              [p["input_vcf"], p["indigenomes"]]),
        ("Stage 2 · Ancestry Inference (PCA)",   [p["input_vcf"], p["kg"]]),
        ("Stage 3 · Pigmentation (HIrisPlex-S)", [p["hiri_csv"], p["hiri_coef"]]),
        ("Stage 4 · Morphology PRS",             [p["gwas_index"]]),
        ("Stage 5 · Face Rendering",             [p["face_templates"]]),
    ]

    stage_rows = []
    for label, deps in stages:
        missing = [d.name for d in deps if not d.exists()]
        stage_rows.append([_ok(not missing), label, "—" if not missing else f"Missing: {', '.join(missing)}"])

    md.table(["", "Stage", "Blocking Issues"], stage_rows)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a comprehensive data audit report."
    )
    parser.add_argument("--config",       default="config.yaml")
    parser.add_argument("--skip-vcf-scan", action="store_true",
                        help="Skip variant counting (fast mode; omits n_variants, rsID overlap)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    cfg_path = root / args.config
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)

    with open(cfg_path) as fh:
        config = yaml.safe_load(fh)

    cp = config["paths"]
    input_vcf_path     = root / cp["input_vcf"]
    indigenomes_path   = root / cp["indigenomes_vcf"]
    kg_path            = root / cp["thousandg_panel"]
    hiri_csv_path      = root / cp["hirisplex_snp_list"]
    hiri_coef_path     = hiri_csv_path.parent / "hirisplex_coefficients.json"
    gwas_dir           = root / cp["gwas_sumstats_dir"]
    gwas_index_path    = root / cp["processed_dir"] / "gwas_trait_index.csv"
    output_dir         = root / cp["output_dir"]
    processed_dir      = root / cp["processed_dir"]
    face_template_dir  = root / cp["face_template_dir"]

    skip = args.skip_vcf_scan

    # ------------------------------------------------------------------
    # Single-pass VCF inspection (collect rsIDs from 1KG in same pass)
    # ------------------------------------------------------------------
    log.info("Inspecting Input VCF …")
    input_info, _ = inspect_vcf(input_vcf_path, skip_scan=skip)

    log.info("Inspecting IndiGenomes VCF …")
    indig_info, _ = inspect_vcf(indigenomes_path, skip_scan=skip)

    log.info("Inspecting 1000 Genomes VCF (collecting rsIDs) …")
    kg_info, kg_ids = inspect_vcf(kg_path, collect_ids=(not skip), skip_scan=skip)
    log.info("1KG rsID set size: %d", len(kg_ids))

    datasets = {
        "Input VCF (subject)":    (input_vcf_path,  input_info),
        "IndiGenomes":            (indigenomes_path, indig_info),
        "1000 Genomes SAS+EUR":   (kg_path,         kg_info),
    }
    builds = {label: info["build"] for label, (_, info) in datasets.items()}
    kg_missing = not kg_path.exists() or bool(kg_info.get("error"))

    hiri_snps = read_hirisplex_snps(hiri_csv_path)

    gwas_files_exist = gwas_dir.exists() and any(
        f for f in gwas_dir.rglob("*") if f.is_file() and _is_sumstat(f)
    )

    # ------------------------------------------------------------------
    # Assemble report
    # ------------------------------------------------------------------
    md = MD()
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    md.raw(f"# Data Audit Report\n")
    md.raw(f"**Generated:** {now}  \n**Pipeline:** DNA Facial Approximation — Indian forensic context\n")

    if skip:
        md.raw("> *Fast mode (`--skip-vcf-scan`): variant counts and rsID overlap are omitted.*\n")

    section_inventory(md, datasets, hiri_snps, hiri_csv_path, gwas_dir)

    section_hirisplex(md, hiri_snps, kg_ids, kg_missing or skip)

    section_gwas(md, gwas_dir, kg_ids, kg_missing or skip)

    section_build(md, builds)

    paths = {
        "input_vcf":      input_vcf_path,
        "indigenomes":    indigenomes_path,
        "kg":             kg_path,
        "hiri_csv":       hiri_csv_path,
        "hiri_coef":      hiri_coef_path,
        "gwas_index":     gwas_index_path,
        "face_templates": face_template_dir,
        "output_dir":     output_dir,
        "processed_dir":  processed_dir,
    }
    gwas_file_count = sum(
        1 for f in gwas_dir.rglob("*") if f.is_file() and _is_sumstat(f)
    ) if gwas_dir.exists() else 0

    section_checklist(md, paths, gwas_file_count)

    # ------------------------------------------------------------------
    # Write report
    # ------------------------------------------------------------------
    out_path = root / "docs" / "data_audit.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md.build(), encoding="utf-8")

    log.info("Report written → %s", out_path)
    print(f"\nAudit report saved to: {out_path}\n")


if __name__ == "__main__":
    main()

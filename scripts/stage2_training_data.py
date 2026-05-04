"""
stage2_training_data.py

Builds training data for stage-2 fine-grained South Asian sub-ancestry
classification.

Data source (priority order)
-----------------------------
1. IndiGenomes VCF (data/raw/indigenomes/indigenomes.vcf.gz) + sidecar
   metadata file with columns: sample/IID  and  population/pop_group.
2. 1000 Genomes SAS samples from the merged panel VCF, with proxy labels:
     BEB → Dravidian-proxy
     GIH → IndoAryan-proxy
     ITU → Dravidian-proxy
     PJL → IndoAryan-proxy
     STU → Dravidian-proxy

Feature selection
-----------------
Pairwise Hudson FST is computed for every population pair using
scikit-allel.  Per-variant mean FST across all pairs ranks SNPs.
Top 5 000 SNPs are retained (configurable via --top-snps).

Outputs  (data/processed/stage2_training/)
------------------------------------------
  features.npy       float32 dosage matrix  (n_samples × n_snps)
  labels.npy         int32 encoded labels   (n_samples,)
  label_encoder.pkl  sklearn LabelEncoder   (decode int → class name)
  snp_list.csv       CHROM, POS, ID, REF, ALT, mean_fst
  metadata.json      run parameters, class map, counts, timestamp

Usage
-----
  python scripts/stage2_training_data.py [OPTIONS]

Options
  --config PATH           config.yaml (default: project root)
  --indigenomes-vcf PATH  override IndiGenomes VCF path
  --thousandg-vcf PATH    override 1KG merged VCF path
  --panel PATH            override 1KG panel TSV path
  --out-dir PATH          override output directory
  --top-snps INT          FST-selected SNPs to keep  (default: 5000)
  --min-samples INT       flag classes below this count  (default: 20)
  --min-maf FLOAT         MAF filter  (default: 0.01)
  --min-call-rate FLOAT   per-variant call rate  (default: 0.95)
  --max-variants INT      max variants fed to FST; random sub-sample
                          if exceeded  (default: 500000)
  --zarr-cache DIR        directory for zarr cache
  --no-cache              force re-conversion of VCF → zarr
  --seed INT              random seed for sub-sampling  (default: 42)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import allel
import joblib
import numpy as np
import pandas as pd
import yaml
import zarr
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAS_PROXY_MAP: dict[str, str] = {
    "BEB": "Dravidian-proxy",
    "GIH": "IndoAryan-proxy",
    "ITU": "Dravidian-proxy",
    "PJL": "IndoAryan-proxy",
    "STU": "Dravidian-proxy",
}

TOP_N_SNPS_DEFAULT    = 5_000
MIN_CLASS_DEFAULT     = 20
MIN_MAF_DEFAULT       = 0.01
MIN_CALL_RATE_DEFAULT = 0.95
MAX_VARIANTS_DEFAULT  = 500_000
ZARR_CHUNK_LENGTH     = 10_000
MIN_VCF_BYTES         = 2_048   # anything smaller is a stub / placeholder


# ===========================================================================
# Data-source detection
# ===========================================================================

def _is_valid_vcf(path: Path) -> bool:
    return path.exists() and path.stat().st_size >= MIN_VCF_BYTES


def _find_indigenomes_metadata(vcf_path: Path) -> Path | None:
    stem = vcf_path.name.split(".")[0]
    candidates = [
        vcf_path.parent / f"{stem}.metadata.tsv",
        vcf_path.parent / f"{stem}.metadata.csv",
        vcf_path.parent / f"{stem}_metadata.tsv",
        vcf_path.parent / f"{stem}_sample_info.tsv",
        vcf_path.parent / "sample_info.tsv",
        vcf_path.parent / "indigenomes_sample_info.tsv",
    ]
    for p in candidates:
        if p.exists():
            log.info("IndiGenomes metadata file: %s", p)
            return p
    return None


def _load_indigenomes_metadata(path: Path) -> pd.DataFrame:
    """
    Load IndiGenomes sample metadata; returns DataFrame(IID, population).

    Accepts flexible column names: sample/iid/sample_id for the ID column,
    population/pop/pop_group/group/ethnicity/subpop for the label column.
    """
    sep = "\t" if path.suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip().lower() for c in df.columns]

    id_aliases  = ("sample", "iid", "sample_id", "id")
    pop_aliases = ("population", "pop", "pop_group", "group", "ethnicity", "subpop")

    for col in id_aliases:
        if col in df.columns:
            df = df.rename(columns={col: "IID"})
            break
    else:
        raise ValueError(
            f"No sample-ID column in {path}. "
            f"Expected one of: {id_aliases}.  Got: {list(df.columns)}"
        )

    for col in pop_aliases:
        if col in df.columns:
            df = df.rename(columns={col: "population"})
            break
    else:
        raise ValueError(
            f"No population column in {path}. "
            f"Expected one of: {pop_aliases}.  Got: {list(df.columns)}"
        )

    return df[["IID", "population"]].dropna()


# ===========================================================================
# VCF → zarr (cached)
# ===========================================================================

def _vcf_to_zarr(
    vcf_path: Path,
    zarr_dir: Path,
    samples: list[str] | None,
    force: bool,
) -> zarr.Group:
    """Convert VCF to zarr store (reuse cache when available)."""
    tag = vcf_path.stem.replace(".vcf", "").replace(".gz", "")
    if samples is not None:
        import hashlib
        h = hashlib.md5(",".join(sorted(samples)).encode()).hexdigest()[:8]
        tag = f"{tag}_{h}"
    zarr_path = zarr_dir / f"{tag}.zarr"

    if zarr_path.exists() and not force:
        log.info("Using cached zarr: %s", zarr_path)
        return zarr.open_group(str(zarr_path), mode="r")

    log.info("Converting VCF → zarr (one-time; may take several minutes): %s", vcf_path)
    zarr_path.parent.mkdir(parents=True, exist_ok=True)

    kw: dict[str, Any] = {
        "input":        str(vcf_path),
        "output":       str(zarr_path),
        "fields":       ["CHROM", "POS", "ID", "REF", "ALT", "calldata/GT"],
        "overwrite":    True,
        "chunk_length": ZARR_CHUNK_LENGTH,
        "log":          sys.stdout,
    }
    if samples is not None:
        kw["samples"] = samples

    allel.vcf_to_zarr(**kw)
    log.info("Zarr ready → %s", zarr_path)
    return zarr.open_group(str(zarr_path), mode="r")


def _load_gt_and_samples(callset: zarr.Group) -> tuple[allel.GenotypeArray, list[str]]:
    """Load full genotype array and decoded sample list from zarr store."""
    raw = callset["calldata/GT"][:]
    gt  = allel.GenotypeArray(raw)
    samples = [
        s.decode() if isinstance(s, bytes) else str(s)
        for s in callset["samples"][:]
    ]
    return gt, samples


def _decode_str_array(arr: np.ndarray) -> np.ndarray:
    """Decode bytes→str in a numpy array (zarr stores strings as bytes)."""
    if arr.dtype.kind in ("S", "O"):
        vdec = np.vectorize(lambda x: x.decode() if isinstance(x, bytes) else str(x))
        return vdec(arr)
    return arr


# ===========================================================================
# QC
# ===========================================================================

def _apply_qc(
    gt: allel.GenotypeArray,
    min_maf: float,
    min_call_rate: float,
) -> np.ndarray:
    """Return boolean variant mask (True = keep)."""
    n_samples = gt.shape[1]

    cr_mask  = (gt.count_called(axis=1) / n_samples) >= min_call_rate
    ac       = gt.count_alleles()
    bi_mask  = ac.is_biallelic()
    af       = ac.to_frequencies()
    maf      = np.min(af[:, :2], axis=1) if af.shape[1] >= 2 else np.zeros(gt.shape[0])
    maf_mask = maf >= min_maf

    mask = cr_mask & bi_mask & maf_mask
    log.info(
        "QC: %d / %d variants pass  "
        "(call_rate>=%.2f: %d | biallelic: %d | MAF>=%.3f: %d)",
        mask.sum(), len(mask),
        min_call_rate, cr_mask.sum(),
        bi_mask.sum(),
        min_maf, maf_mask.sum(),
    )
    return mask


# ===========================================================================
# FST
# ===========================================================================

def _compute_mean_pairwise_fst(
    gt: allel.GenotypeArray,
    subpop_indices: list[np.ndarray],
    pop_names: list[str],
) -> np.ndarray:
    """
    Hudson FST for every population pair.
    Returns per-variant mean FST across all pairs (NaN → 0).
    """
    pairs = list(combinations(range(len(pop_names)), 2))
    log.info(
        "FST: %d variant × %d population pairs …",
        gt.shape[0], len(pairs),
    )

    pair_fsts: list[np.ndarray] = []
    for i, j in pairs:
        idx_i, idx_j = subpop_indices[i], subpop_indices[j]
        if len(idx_i) < 2 or len(idx_j) < 2:
            log.warning(
                "Skipping pair (%s, %s): too few samples (%d, %d)",
                pop_names[i], pop_names[j], len(idx_i), len(idx_j),
            )
            continue

        ac_i = gt[:, idx_i, :].count_alleles()
        ac_j = gt[:, idx_j, :].count_alleles()
        num, den = allel.hudson_fst(ac_i, ac_j)

        with np.errstate(invalid="ignore", divide="ignore"):
            fst = np.where(den > 0, num / den, np.nan)

        log.info(
            "  FST(%s, %s): mean=%.4f  valid=%d / %d",
            pop_names[i], pop_names[j],
            float(np.nanmean(fst)),
            int(np.isfinite(fst).sum()), len(fst),
        )
        pair_fsts.append(fst)

    if not pair_fsts:
        raise RuntimeError(
            "No valid population pairs for FST. "
            "Ensure at least 2 populations have ≥ 2 samples each."
        )

    mean_fst = np.nanmean(np.stack(pair_fsts, axis=1), axis=1)
    return np.nan_to_num(mean_fst, nan=0.0)


# ===========================================================================
# Feature selection & dosage
# ===========================================================================

def _select_top_snps(mean_fst: np.ndarray, n: int) -> np.ndarray:
    """Indices of top-n variants by mean FST, sorted descending."""
    n_avail = len(mean_fst)
    if n_avail <= n:
        log.warning(
            "Only %d variants available; using all (requested top-%d).",
            n_avail, n,
        )
        return np.argsort(mean_fst)[::-1]
    top_idx = np.argsort(mean_fst)[::-1][:n]
    log.info(
        "Selected top %d SNPs  FST [%.4f – %.4f]",
        n, float(mean_fst[top_idx[-1]]), float(mean_fst[top_idx[0]]),
    )
    return top_idx


def _extract_dosage(gt: allel.GenotypeArray, snp_idx: np.ndarray) -> np.ndarray:
    """
    Alt-allele dosage matrix (n_samples × n_snps) as float32.
    Missing calls are mean-imputed per SNP.
    """
    gt_sub    = gt[snp_idx]                              # (n_snps, n_samples, 2)
    is_miss   = gt_sub[:, :, 0] == -1
    dosage    = gt_sub.sum(axis=2).astype(np.float32)   # 0 / 1 / 2  (-2 for missing)
    dosage[is_miss] = np.nan

    snp_mean = np.nanmean(dosage, axis=1, keepdims=True)
    dosage   = np.where(np.isnan(dosage), snp_mean, dosage)
    dosage   = dosage.T                                  # → (n_samples, n_snps)

    log.info(
        "Dosage: %d samples × %d SNPs  (%.1f%% imputed)",
        dosage.shape[0], dosage.shape[1], 100 * is_miss.mean(),
    )
    return dosage


# ===========================================================================
# Class balance report
# ===========================================================================

def _print_class_balance(
    labels_raw: np.ndarray,
    min_n: int,
) -> None:
    unique, counts = np.unique(labels_raw, return_counts=True)
    total = len(labels_raw)
    flagged = False

    print("\n" + "=" * 64)
    print("  Class balance report")
    print("=" * 64)
    print(f"  {'Population':<32} {'N':>6}  {'%':>6}  Status")
    print("  " + "-" * 60)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        pct  = 100 * cnt / total
        flag = "LOW ⚠" if cnt < min_n else "OK"
        if cnt < min_n:
            flagged = True
        print(f"  {cls:<32} {cnt:>6}  {pct:>5.1f}%  {flag}")
    print("  " + "-" * 60)
    print(f"  {'TOTAL':<32} {total:>6}")
    print("=" * 64)
    if flagged:
        log.warning(
            "Classes below %d samples detected. "
            "Consider merging or augmenting rare classes.",
            min_n,
        )


# ===========================================================================
# Save outputs
# ===========================================================================

def _save_outputs(
    out_dir: Path,
    dosage: np.ndarray,
    labels_enc: np.ndarray,
    le: LabelEncoder,
    snp_df: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "features.npy", dosage)
    np.save(out_dir / "labels.npy", labels_enc)
    joblib.dump(le, out_dir / "label_encoder.pkl", compress=3)
    snp_df.to_csv(out_dir / "snp_list.csv", index=False)
    with open(out_dir / "metadata.json", "w") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    print(f"\n  Outputs → {out_dir}")
    print(f"    features.npy      {dosage.shape}  float32")
    print(f"    labels.npy        {labels_enc.shape}  int32")
    print(f"    label_encoder.pkl classes={list(le.classes_)}")
    print(f"    snp_list.csv      {len(snp_df)} SNPs")
    print(f"    metadata.json")


# ===========================================================================
# Shared processing pipeline
# ===========================================================================

def _run_pipeline(
    *,
    gt: allel.GenotypeArray,
    zarr_samples: list[str],
    pop_map: dict[str, str],
    callset: zarr.Group,
    min_maf: float,
    min_call_rate: float,
    max_variants: int,
    top_snps: int,
    min_samples: int,
    seed: int,
    out_dir: Path,
    data_source: str,
    data_source_path: str,
    proxy_map: dict[str, str] | None,
) -> None:
    rng = np.random.default_rng(seed)

    # Align samples: only those present in both zarr and pop_map
    valid_samples = [s for s in zarr_samples if s in pop_map]
    if not valid_samples:
        log.error("No overlap between VCF samples and population map. Check IDs.")
        sys.exit(1)
    sample_order = {s: i for i, s in enumerate(zarr_samples)}
    col_idx = np.array([sample_order[s] for s in valid_samples])

    gt_filt = allel.GenotypeArray(gt[:, col_idx, :])
    labels_raw = np.array([pop_map[s] for s in valid_samples])

    log.info(
        "Working set: %d samples, %d variants",
        gt_filt.shape[1], gt_filt.shape[0],
    )

    # ---- QC ----
    qc_mask           = _apply_qc(gt_filt, min_maf, min_call_rate)
    gt_qc             = allel.GenotypeArray(gt_filt[qc_mask])
    qc_original_idx   = np.where(qc_mask)[0]   # maps QC pos → original pos

    # ---- Sub-sample for FST if too many variants ----
    n_qc = gt_qc.shape[0]
    if n_qc > max_variants:
        log.info(
            "%d variants after QC; random sub-sample to %d for FST.",
            n_qc, max_variants,
        )
        chosen        = np.sort(rng.choice(n_qc, size=max_variants, replace=False))
        gt_fst        = allel.GenotypeArray(gt_qc[chosen])
        fst_in_qc_idx = chosen                  # maps FST pos → QC pos
    else:
        gt_fst        = gt_qc
        fst_in_qc_idx = np.arange(n_qc)

    # ---- Population sub-arrays for FST ----
    classes        = sorted(np.unique(labels_raw).tolist())
    subpop_indices = [np.where(labels_raw == cls)[0] for cls in classes]

    # ---- FST ----
    mean_fst = _compute_mean_pairwise_fst(gt_fst, subpop_indices, classes)

    # ---- Top-N selection (indices into gt_fst / mean_fst) ----
    top_fst_idx    = _select_top_snps(mean_fst, top_snps)
    top_fst_values = mean_fst[top_fst_idx]

    # Map top_fst_idx → QC array positions
    top_qc_idx = fst_in_qc_idx[top_fst_idx]

    # Map top_qc_idx → original (pre-QC) variant positions (for SNP metadata)
    top_orig_idx = qc_original_idx[top_qc_idx]

    # ---- Dosage from QC array ----
    dosage = _extract_dosage(gt_qc, top_qc_idx)

    # ---- SNP metadata ----
    chrom = _decode_str_array(callset["variants/CHROM"][:])
    pos   = callset["variants/POS"][:]
    ids   = _decode_str_array(callset["variants/ID"][:])
    ref   = _decode_str_array(callset["variants/REF"][:])
    alt   = callset["variants/ALT"][:]   # shape (n_variants, n_alt_alleles)

    def _first_alt(row: np.ndarray) -> str:
        if hasattr(row, "__len__") and len(row) > 0:
            v = row[0]
            return v.decode() if isinstance(v, bytes) else str(v)
        return "."

    snp_df = pd.DataFrame({
        "CHROM":    chrom[top_orig_idx],
        "POS":      pos[top_orig_idx],
        "ID":       ids[top_orig_idx],
        "REF":      ref[top_orig_idx],
        "ALT":      [_first_alt(alt[i]) for i in top_orig_idx],
        "mean_fst": top_fst_values,
    }).sort_values("mean_fst", ascending=False).reset_index(drop=True)

    # ---- Label encoding ----
    le          = LabelEncoder()
    labels_enc  = le.fit_transform(labels_raw).astype(np.int32)

    # ---- Class balance ----
    _print_class_balance(labels_raw, min_samples)

    # ---- Metadata ----
    class_counts = {cls: int((labels_raw == cls).sum()) for cls in classes}
    metadata: dict[str, Any] = {
        "data_source":        data_source,
        "data_source_path":   data_source_path,
        "proxy_map":          proxy_map,
        "n_samples":          int(dosage.shape[0]),
        "n_snps":             int(dosage.shape[1]),
        "class_map":          {int(i): c for i, c in enumerate(le.classes_)},
        "class_counts":       class_counts,
        "n_variants_raw":     int(gt_filt.shape[0]),
        "n_variants_qc":      int(n_qc),
        "n_variants_fst":     int(gt_fst.shape[0]),
        "qc_min_maf":         min_maf,
        "qc_min_call_rate":   min_call_rate,
        "top_n_snps":         top_snps,
        "fst_method":         "hudson_pairwise_mean",
        "fst_range_selected": [float(top_fst_values.min()), float(top_fst_values.max())],
        "timestamp":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    _save_outputs(out_dir, dosage, labels_enc, le, snp_df, metadata)


# ===========================================================================
# IndiGenomes path
# ===========================================================================

def _run_indigenomes(
    vcf_path: Path,
    zarr_cache: Path,
    force_zarr: bool,
    **kwargs: Any,
) -> None:
    meta_path = _find_indigenomes_metadata(vcf_path)
    if meta_path is None:
        log.error(
            "IndiGenomes VCF found but no metadata file detected. "
            "Create a TSV with columns (sample/IID) and (population/pop_group) "
            "alongside the VCF: %s",
            vcf_path.parent,
        )
        sys.exit(1)

    meta    = _load_indigenomes_metadata(meta_path)
    pop_map = dict(zip(meta["IID"], meta["population"]))

    log.info(
        "IndiGenomes: %d samples, %d populations: %s",
        len(meta),
        meta["population"].nunique(),
        dict(meta["population"].value_counts()),
    )

    callset           = _vcf_to_zarr(vcf_path, zarr_cache, samples=list(pop_map), force=force_zarr)
    gt, zarr_samples  = _load_gt_and_samples(callset)

    _run_pipeline(
        gt=gt,
        zarr_samples=zarr_samples,
        pop_map=pop_map,
        callset=callset,
        data_source="IndiGenomes",
        data_source_path=str(vcf_path),
        proxy_map=None,
        **kwargs,
    )


# ===========================================================================
# 1000 Genomes SAS fallback path
# ===========================================================================

def _run_1kg_fallback(
    vcf_path: Path,
    panel_path: Path,
    zarr_cache: Path,
    force_zarr: bool,
    **kwargs: Any,
) -> None:
    if not panel_path.exists():
        log.error(
            "1KG panel metadata not found: %s. "
            "Run scripts/download_1kg_sas.sh first.",
            panel_path,
        )
        sys.exit(1)

    panel = pd.read_csv(panel_path, sep="\t", usecols=["sample", "pop", "super_pop"])
    panel = panel.rename(columns={"sample": "IID"})

    sas = panel[panel["super_pop"] == "SAS"].copy()
    unknown_pops = set(sas["pop"].unique()) - set(SAS_PROXY_MAP)
    if unknown_pops:
        log.warning(
            "Unknown SAS population codes not in proxy map: %s "
            "— labelled 'Other-SAS'.",
            unknown_pops,
        )
    sas["population"] = sas["pop"].map(SAS_PROXY_MAP).fillna("Other-SAS")
    pop_map = dict(zip(sas["IID"], sas["population"]))

    log.info(
        "1KG SAS fallback: %d samples  proxy distribution: %s",
        len(sas),
        dict(sas["population"].value_counts()),
    )

    callset           = _vcf_to_zarr(vcf_path, zarr_cache, samples=list(pop_map), force=force_zarr)
    gt, zarr_samples  = _load_gt_and_samples(callset)

    _run_pipeline(
        gt=gt,
        zarr_samples=zarr_samples,
        pop_map=pop_map,
        callset=callset,
        data_source="1000G-SAS-fallback",
        data_source_path=str(vcf_path),
        proxy_map=SAS_PROXY_MAP,
        **kwargs,
    )


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build stage-2 South Asian sub-ancestry training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",          default="config.yaml")
    p.add_argument("--indigenomes-vcf", default=None)
    p.add_argument("--thousandg-vcf",   default=None)
    p.add_argument("--panel",           default=None, help="1KG panel TSV")
    p.add_argument("--out-dir",         default=None)
    p.add_argument("--top-snps",        type=int,   default=TOP_N_SNPS_DEFAULT)
    p.add_argument("--min-samples",     type=int,   default=MIN_CLASS_DEFAULT)
    p.add_argument("--min-maf",         type=float, default=MIN_MAF_DEFAULT)
    p.add_argument("--min-call-rate",   type=float, default=MIN_CALL_RATE_DEFAULT)
    p.add_argument("--max-variants",    type=int,   default=MAX_VARIANTS_DEFAULT,
                   help="Max variants used for FST; random sub-sample if exceeded")
    p.add_argument("--zarr-cache",      default=None)
    p.add_argument("--no-cache",        action="store_true",
                   help="Force VCF → zarr re-conversion")
    p.add_argument("--seed",            type=int, default=42)
    return p.parse_args()


def _resolve(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    cfg_path = root / args.config
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)
    p = cfg["paths"]

    return {
        "indigenomes_vcf": (
            Path(args.indigenomes_vcf) if args.indigenomes_vcf
            else root / p["indigenomes_vcf"]
        ),
        "thousandg_vcf": (
            Path(args.thousandg_vcf) if args.thousandg_vcf
            else root / p["thousandg_panel"]
        ),
        "panel": (
            Path(args.panel) if args.panel
            else root / "data/raw/1kg/integrated_call_samples_v3.panel"
        ),
        "out_dir": (
            Path(args.out_dir) if args.out_dir
            else root / p["processed_dir"] / "stage2_training"
        ),
        "zarr_cache": (
            Path(args.zarr_cache) if args.zarr_cache
            else root / p["processed_dir"] / "zarr_cache"
        ),
    }


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> None:
    args  = _parse_args()
    root  = Path(__file__).resolve().parents[1]
    paths = _resolve(args, root)

    indigenomes_vcf = paths["indigenomes_vcf"]
    thousandg_vcf   = paths["thousandg_vcf"]

    use_indigenomes = _is_valid_vcf(indigenomes_vcf)

    if use_indigenomes:
        log.info("IndiGenomes VCF found — using as primary data source.")
    else:
        if not _is_valid_vcf(thousandg_vcf):
            log.error(
                "No usable data source found.\n"
                "  IndiGenomes VCF: %s\n"
                "  1KG merged VCF : %s\n"
                "Run scripts/download_indigenomes.py or scripts/download_1kg_sas.sh.",
                indigenomes_vcf, thousandg_vcf,
            )
            sys.exit(1)
        log.info(
            "IndiGenomes VCF not available — falling back to 1KG SAS samples. "
            "Proxy map: %s",
            SAS_PROXY_MAP,
        )

    common = dict(
        min_maf=args.min_maf,
        min_call_rate=args.min_call_rate,
        max_variants=args.max_variants,
        top_snps=args.top_snps,
        min_samples=args.min_samples,
        seed=args.seed,
        out_dir=paths["out_dir"],
    )

    if use_indigenomes:
        _run_indigenomes(
            vcf_path=indigenomes_vcf,
            zarr_cache=paths["zarr_cache"],
            force_zarr=args.no_cache,
            **common,
        )
    else:
        _run_1kg_fallback(
            vcf_path=thousandg_vcf,
            panel_path=paths["panel"],
            zarr_cache=paths["zarr_cache"],
            force_zarr=args.no_cache,
            **common,
        )


if __name__ == "__main__":
    main()

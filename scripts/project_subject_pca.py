"""
project_subject_pca.py

Projects a forensic subject's VCF into the PCA space built by pca_prep.sh,
then saves combined PC scores and generates a population-stratification plot.

Pipeline
--------
1. Load the LD-pruned SNP manifest (step5_pruned.pvar) — ordered list of
   (CHROM, POS, ID, REF, ALT) for the ~n-thousand pruned markers.
2. Scan the subject VCF with cyvcf2, extracting dosages (0/1/2) at those
   positions; flip strand when REF/ALT are swapped vs the reference panel.
3. Impute SNPs missing from the subject VCF with 2 × ALT_FREQ (dosage mean
   under HWE, derived from step6_pca.afreq).
4. Standardise the dosage vector: z = (g − 2p) / sqrt(2p(1−p)), clamping
   loci with zero variance to 0.
5. Project into PC space: score = z @ V, where V (n_snps × 20) is the
   right-singular-vector matrix from step6_pca.eigenvec.var.
6. Append the subject row to the reference eigenvec table and save as
   {output_dir}/subject_pca_scores.csv.
7. Generate scatter plots (PC1×PC2, PC1×PC3) saved as
   {output_dir}/pca_projection_PC1_PC2.png  and
   {output_dir}/pca_projection_PC1_PC3.png.

Required input files (all produced by pca_prep.sh)
---------------------------------------------------
  data/processed/pca/step5_pruned.pvar       SNP list + alleles
  data/processed/pca/step6_pca.eigenvec      reference sample PC scores
  data/processed/pca/step6_pca.eigenvec.var  variant PC loadings
  data/processed/pca/step6_pca.afreq         reference ALT allele frequencies
  data/processed/pca/step6_pca.eigenval      eigenvalues (variance explained)
  data/raw/1kg/integrated_call_samples_v3.panel  population annotations

Usage
-----
  python scripts/project_subject_pca.py subject.vcf.gz [OPTIONS]

Options
  --config PATH     config.yaml (default: config.yaml in project root)
  --pca-dir PATH    override PCA output directory
  --panel PATH      override 1KG panel file
  --output-dir PATH override output directory
  --pc-pairs STR    comma-separated PC pair(s) to plot, e.g. "1,2 1,3" (default: "1,2 1,3")
  --subject-id STR  label for the subject sample (default: derived from VCF filename)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import yaml

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Population colour palette (colourblind-safe)
# ---------------------------------------------------------------------------
SUPERPOP_COLOURS = {
    "SAS": "#4e79a7",   # steel blue
    "EUR": "#f28e2b",   # amber
    "AFR": "#59a14f",   # green  (not expected here, but defensive)
    "EAS": "#e15759",   # rose
    "AMR": "#76b7b2",   # teal
    "UNKNOWN": "#bab0ac",
}
SUBJECT_COLOUR  = "#d62728"   # bright red
SUBJECT_MARKER  = "*"
SUBJECT_SIZE    = 400
REF_ALPHA       = 0.5
REF_SIZE        = 18


# ===========================================================================
# I/O helpers
# ===========================================================================

def _tsv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a PLINK2 tab-separated file; strips the leading # from the header."""
    df = pd.read_csv(path, sep="\t", comment=None, **kwargs)
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def load_pvar(pvar_path: Path) -> pd.DataFrame:
    """
    Load step5_pruned.pvar.

    Returns DataFrame with columns CHROM, POS, ID, REF, ALT
    and a positional integer index (row order = PCA loading order).
    """
    df = _tsv(pvar_path, dtype={"CHROM": str, "POS": int, "ID": str,
                                "REF": str, "ALT": str})
    df["CHROM"] = df["CHROM"].str.lstrip("chr")
    df = df.reset_index(drop=True)
    log.info("Loaded pvar: %d SNPs", len(df))
    return df


def load_eigenvec_var(path: Path, n_pcs: int) -> np.ndarray:
    """
    Load step6_pca.eigenvec.var.

    Returns ndarray shape (n_snps, n_pcs) aligned to pvar row order.
    """
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    df = _tsv(path, dtype={"CHROM": str, "POS": int})
    present = [c for c in pc_cols if c in df.columns]
    if not present:
        raise ValueError(
            f"No PC columns found in {path}. "
            "Ensure pca_prep.sh was run with '--pca N approx var-wts'."
        )
    V = df[present].to_numpy(dtype=np.float64)
    log.info("Loaded variant loadings: %d SNPs × %d PCs", *V.shape)
    return V


def load_afreq(path: Path) -> np.ndarray:
    """
    Load step6_pca.afreq.

    Returns ALT_FREQS aligned to pvar row order (same pgen → same row order).
    """
    df = _tsv(path, dtype={"CHROM": str, "ID": str,
                            "REF": str, "ALT": str,
                            "ALT_FREQS": float, "OBS_CT": int})
    freqs = df["ALT_FREQS"].to_numpy(dtype=np.float64)
    log.info("Loaded allele frequencies: %d SNPs", len(freqs))
    return freqs


def load_eigenvec(path: Path) -> pd.DataFrame:
    """
    Load step6_pca.eigenvec.

    Returns DataFrame with columns IID, PC1 … PC20.
    """
    df = _tsv(path)
    if "IID" not in df.columns and "FID" in df.columns:
        df = df.rename(columns={"FID": "IID_DROP"})
    if "#FID" in df.columns:
        df = df.drop(columns=["#FID"], errors="ignore")
    # eigenvec has FID IID PC1 … after stripping '#'
    if "FID" in df.columns:
        df = df.drop(columns=["FID"])
    log.info("Loaded reference eigenvec: %d samples", len(df))
    return df


def load_eigenval(path: Path) -> np.ndarray:
    return np.loadtxt(path)


def load_panel(path: Path) -> pd.DataFrame:
    """
    Load 1KG panel file (sample pop super_pop gender).

    Returns DataFrame with columns sample, pop, super_pop.
    """
    df = pd.read_csv(path, sep="\t", usecols=["sample", "pop", "super_pop"])
    return df


# ===========================================================================
# Genotype extraction from subject VCF
# ===========================================================================

def _parse_dosage(var, sample_idx: int,
                  ref: str, alt: str) -> Optional[float]:
    """
    Extract dosage (0/1/2) for one variant from a cyvcf2 Variant.

    Handles REF/ALT swap vs the reference panel; returns None on mismatch.
    """
    gt = var.genotypes[sample_idx]   # [allele1, allele2, phased_bool]
    a1, a2 = gt[0], gt[1]
    if a1 < 0 or a2 < 0:            # missing genotype
        return None

    vcf_ref = var.REF
    vcf_alts = var.ALT               # tuple of alt alleles

    # Build per-allele dosage counts: allele index 0 = REF, 1 = first ALT, etc.
    allele_indices = [a1, a2]

    # Determine what allele (0/1) each called index corresponds to in our panel
    # Our panel: REF = ref_panel_ref, ALT = ref_panel_alt (biallelic, first ALT)
    def panel_dosage(idx: int) -> Optional[int]:
        if idx == 0:
            vcf_allele = vcf_ref
        elif idx <= len(vcf_alts):
            vcf_allele = vcf_alts[idx - 1]
        else:
            return None

        if vcf_allele == ref:
            return 0
        if vcf_allele == alt:
            return 1
        # Complement check (handle strand flip)
        complement = {"A": "T", "T": "A", "C": "G", "G": "C"}
        comp_allele = "".join(complement.get(b, "N") for b in vcf_allele)
        if comp_allele == ref:
            return 0
        if comp_allele == alt:
            return 1
        return None

    d1 = panel_dosage(a1)
    d2 = panel_dosage(a2)
    if d1 is None or d2 is None:
        return None
    return float(d1 + d2)


def extract_subject_dosages(
    subject_vcf: Path,
    pvar: pd.DataFrame,
    subject_sample_idx: int = 0,
) -> np.ndarray:
    """
    Scan subject VCF and extract dosages at every position in pvar.

    Returns ndarray shape (n_snps,) with np.nan for missing positions.
    """
    try:
        import cyvcf2
    except ImportError:
        log.error("cyvcf2 is required: pip install cyvcf2")
        sys.exit(1)

    # Build lookup: (normalised_chrom, pos) → (pvar_row_idx, ref, alt)
    target: dict[tuple[str, int], tuple[int, str, str]] = {}
    for i, row in pvar.iterrows():
        target[(row["CHROM"], row["POS"])] = (i, row["REF"], row["ALT"])

    dosages = np.full(len(pvar), np.nan, dtype=np.float64)
    n_found = 0

    log.info("Scanning subject VCF: %s …", subject_vcf)
    with cyvcf2.VCF(str(subject_vcf)) as vcf:
        if len(vcf.samples) == 0:
            raise ValueError("Subject VCF has no samples.")
        if subject_sample_idx >= len(vcf.samples):
            raise ValueError(
                f"Sample index {subject_sample_idx} out of range "
                f"(VCF has {len(vcf.samples)} sample(s))."
            )

        for var in vcf:
            chrom = var.CHROM.lstrip("chr")
            key = (chrom, var.POS)
            if key not in target:
                continue
            row_idx, ref, alt = target[key]
            d = _parse_dosage(var, subject_sample_idx, ref, alt)
            if d is not None:
                dosages[row_idx] = d
                n_found += 1

    n_total = len(pvar)
    n_miss = n_total - n_found
    log.info(
        "Genotypes extracted: %d/%d SNPs found (%.1f%% missing, will be imputed)",
        n_found, n_total, 100 * n_miss / n_total,
    )
    if n_found < 0.5 * n_total:
        log.warning(
            "Fewer than 50%% of pruned SNPs found in subject VCF. "
            "Check that the VCF uses the same genome build as the reference panel."
        )
    return dosages


# ===========================================================================
# Imputation, standardisation, projection
# ===========================================================================

def impute_missing(dosages: np.ndarray, ref_af: np.ndarray) -> np.ndarray:
    """Replace np.nan with 2 × ALT_FREQ (Hardy-Weinberg dosage mean)."""
    g = dosages.copy()
    missing_mask = np.isnan(g)
    g[missing_mask] = 2.0 * ref_af[missing_mask]
    log.info("Imputed %d missing SNPs with 2×AF", missing_mask.sum())
    return g


def standardise(g: np.ndarray, ref_af: np.ndarray) -> np.ndarray:
    """
    Standardise dosage vector: z = (g − 2p) / sqrt(2p(1−p)).

    Loci with zero variance (monomorphic in reference) are set to 0.
    """
    mean = 2.0 * ref_af
    var  = 2.0 * ref_af * (1.0 - ref_af)
    sd   = np.sqrt(var)

    zero_var = sd == 0.0
    if zero_var.any():
        log.debug("Setting %d zero-variance loci to 0 before standardisation",
                  zero_var.sum())
        sd[zero_var] = 1.0          # will produce 0 after subtraction

    z = (g - mean) / sd
    z[zero_var] = 0.0
    return z


def project(z: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """
    Project standardised dosage vector onto PC space.

    z        : shape (n_snps,)
    loadings : shape (n_snps, n_pcs)
    returns  : shape (n_pcs,)
    """
    return z @ loadings


# ===========================================================================
# Output: combined CSV
# ===========================================================================

def build_combined_df(
    eigenvec: pd.DataFrame,
    panel: pd.DataFrame,
    subject_scores: np.ndarray,
    subject_id: str,
    n_pcs: int,
) -> pd.DataFrame:
    """
    Merge reference eigenvec with population labels, append subject row.
    """
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]

    ref_df = eigenvec.copy()
    if "IID" not in ref_df.columns:
        raise ValueError("eigenvec DataFrame must have an 'IID' column.")

    ref_df = ref_df.merge(
        panel.rename(columns={"sample": "IID"}),
        on="IID", how="left",
    )
    ref_df["super_pop"] = ref_df["super_pop"].fillna("UNKNOWN")
    ref_df["pop"]       = ref_df["pop"].fillna("UNKNOWN")
    ref_df["is_subject"] = False

    subject_row = {col: val for col, val in zip(pc_cols, subject_scores)}
    subject_row.update({
        "IID": subject_id,
        "super_pop": "SUBJECT",
        "pop": "SUBJECT",
        "is_subject": True,
    })
    subject_df = pd.DataFrame([subject_row])

    combined = pd.concat([ref_df, subject_df], ignore_index=True)
    return combined


def save_csv(combined: pd.DataFrame, out_path: Path, n_pcs: int) -> None:
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    cols = ["IID", "super_pop", "pop", "is_subject"] + pc_cols
    combined[cols].to_csv(out_path, index=False)
    log.info("Combined PC scores saved → %s", out_path)


# ===========================================================================
# Plotting
# ===========================================================================

def _variance_explained(eigenval: np.ndarray) -> list[float]:
    total = eigenval.sum()
    return [ev / total * 100 for ev in eigenval]


def plot_pca(
    combined: pd.DataFrame,
    pc_x: int,
    pc_y: int,
    eigenval: np.ndarray,
    subject_id: str,
    out_path: Path,
) -> None:
    var_exp = _variance_explained(eigenval)
    x_col, y_col = f"PC{pc_x}", f"PC{pc_y}"

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_facecolor("#f8f8f8")
    ax.grid(True, color="white", linewidth=0.8, zorder=0)

    # ---- Reference samples (grouped by superpopulation) --------------------
    ref = combined[~combined["is_subject"]]
    for spop, grp in ref.groupby("super_pop"):
        colour = SUPERPOP_COLOURS.get(spop, SUPERPOP_COLOURS["UNKNOWN"])
        ax.scatter(
            grp[x_col], grp[y_col],
            c=colour, s=REF_SIZE, alpha=REF_ALPHA,
            label=spop, linewidths=0, zorder=2,
        )

    # ---- Subject sample ----------------------------------------------------
    subj = combined[combined["is_subject"]]
    if not subj.empty:
        sx, sy = subj[x_col].iloc[0], subj[y_col].iloc[0]
        ax.scatter(
            sx, sy,
            c=SUBJECT_COLOUR, s=SUBJECT_SIZE,
            marker=SUBJECT_MARKER, zorder=5,
            edgecolors="black", linewidths=0.8,
            label=f"Subject ({subject_id})",
        )
        # Annotation with drop-shadow effect
        ax.annotate(
            subject_id,
            (sx, sy),
            xytext=(8, 8), textcoords="offset points",
            fontsize=9, fontweight="bold", color=SUBJECT_COLOUR,
            path_effects=[
                pe.withStroke(linewidth=2.5, foreground="white")
            ],
            zorder=6,
        )

    ax.set_xlabel(
        f"PC{pc_x}  ({var_exp[pc_x - 1]:.2f}% variance explained)",
        fontsize=11,
    )
    ax.set_ylabel(
        f"PC{pc_y}  ({var_exp[pc_y - 1]:.2f}% variance explained)",
        fontsize=11,
    )
    ax.set_title(
        f"PCA projection — PC{pc_x} vs PC{pc_y}\n"
        f"Reference: 1000 Genomes SAS + EUR  |  Subject: {subject_id}",
        fontsize=12, pad=12,
    )

    legend = ax.legend(
        title="Super-population",
        fontsize=9, title_fontsize=9,
        markerscale=1.4, framealpha=0.9,
        loc="upper right",
    )
    legend.get_frame().set_linewidth(0.5)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved → %s", out_path)


def plot_variance_bar(
    eigenval: np.ndarray,
    out_path: Path,
    n_show: int = 20,
) -> None:
    """Scree-style bar chart of variance explained per PC."""
    var_exp = _variance_explained(eigenval)[:n_show]
    cumulative = np.cumsum(var_exp)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(1, len(var_exp) + 1)
    ax.bar(x, var_exp, color="#4e79a7", alpha=0.8, label="Var. explained")
    ax.plot(x, cumulative, color="#f28e2b", marker="o",
            markersize=5, linewidth=1.5, label="Cumulative")
    ax.set_xlabel("PC", fontsize=11)
    ax.set_ylabel("Variance explained (%)", fontsize=11)
    ax.set_title("PCA scree plot — reference panel", fontsize=12)
    ax.set_xticks(x)
    ax.legend(fontsize=9)
    ax.grid(axis="y", color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Scree plot saved → %s", out_path)


# ===========================================================================
# CLI
# ===========================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Project a subject VCF into the 1KG reference PCA space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("subject_vcf", help="Subject genotype VCF (or VCF.gz)")
    p.add_argument("--config",       default="config.yaml")
    p.add_argument("--pca-dir",      default=None,
                   help="Override PCA directory (default: data/processed/pca/)")
    p.add_argument("--panel",        default=None,
                   help="Override 1KG panel file path")
    p.add_argument("--output-dir",   default=None,
                   help="Override output directory (default: outputs/)")
    p.add_argument("--pc-pairs",     default="1,2 1,3",
                   help="Space-separated PC pairs to plot, e.g. '1,2 1,3'")
    p.add_argument("--subject-id",   default=None,
                   help="Label for subject (default: VCF filename stem)")
    p.add_argument("--sample-index", type=int, default=0,
                   help="Index of the sample to project when VCF has multiple samples (default: 0)")
    return p.parse_args()


def _resolve_paths(args: argparse.Namespace, root: Path) -> dict[str, Path]:
    with open(root / args.config) as fh:
        cfg = yaml.safe_load(fh)

    pca_dir    = Path(args.pca_dir)    if args.pca_dir    else root / cfg["paths"]["processed_dir"] / "pca"
    output_dir = Path(args.output_dir) if args.output_dir else root / cfg["paths"]["output_dir"]
    panel      = Path(args.panel)      if args.panel      else root / "data/raw/1kg/integrated_call_samples_v3.panel"

    return {
        "pca_dir":    pca_dir,
        "output_dir": output_dir,
        "panel":      panel,
        "pvar":       pca_dir / "step5_pruned.pvar",
        "eigenvec":   pca_dir / "step6_pca.eigenvec",
        "eigenvec_var": pca_dir / "step6_pca.eigenvec.var",
        "afreq":      pca_dir / "step6_pca.afreq",
        "eigenval":   pca_dir / "step6_pca.eigenval",
    }


def _check_files(paths: dict[str, Path]) -> None:
    required = {
        "pvar":         "step5_pruned.pvar  — run pca_prep.sh first",
        "eigenvec":     "step6_pca.eigenvec — run pca_prep.sh first",
        "eigenvec_var": "step6_pca.eigenvec.var — run pca_prep.sh with '--pca N approx var-wts'",
        "afreq":        "step6_pca.afreq — run pca_prep.sh (step 6b computes this via --freq)",
        "eigenval":     "step6_pca.eigenval — run pca_prep.sh first",
    }
    missing = [f"{key}: {hint}" for key, hint in required.items()
               if not paths[key].exists()]
    if missing:
        log.error("Required input files not found:\n  %s", "\n  ".join(missing))
        sys.exit(1)


def main() -> None:
    args  = parse_args()
    root  = Path(__file__).resolve().parents[1]
    paths = _resolve_paths(args, root)

    _check_files(paths)
    paths["output_dir"].mkdir(parents=True, exist_ok=True)

    subject_vcf = Path(args.subject_vcf)
    if not subject_vcf.exists():
        log.error("Subject VCF not found: %s", subject_vcf)
        sys.exit(1)

    subject_id = args.subject_id or subject_vcf.stem.split(".")[0]

    # ------------------------------------------------------------------
    # Load reference PCA infrastructure
    # ------------------------------------------------------------------
    log.info("Loading PCA reference files …")
    pvar         = load_pvar(paths["pvar"])
    n_pcs        = sum(1 for c in pd.read_csv(paths["eigenvec"], sep="\t", nrows=0)
                       if c.startswith("PC"))
    loadings     = load_eigenvec_var(paths["eigenvec_var"], n_pcs)
    ref_af       = load_afreq(paths["afreq"])
    eigenvec     = load_eigenvec(paths["eigenvec"])
    eigenval     = load_eigenval(paths["eigenval"])

    # Defensive shape checks
    n_snps = len(pvar)
    if loadings.shape[0] != n_snps:
        log.error(
            "Loading matrix has %d rows but pvar has %d SNPs. "
            "These files must come from the same pca_prep.sh run.",
            loadings.shape[0], n_snps,
        )
        sys.exit(1)
    if len(ref_af) != n_snps:
        log.error("afreq has %d entries but pvar has %d SNPs.", len(ref_af), n_snps)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Load population panel (optional — warn if missing)
    # ------------------------------------------------------------------
    if paths["panel"].exists():
        panel = load_panel(paths["panel"])
    else:
        log.warning(
            "Population panel not found: %s\n"
            "  Reference samples will be plotted without population labels.\n"
            "  Run scripts/download_1kg_sas.sh to create it.",
            paths["panel"],
        )
        panel = pd.DataFrame(columns=["sample", "pop", "super_pop"])

    # ------------------------------------------------------------------
    # Step 1-5: extract → impute → standardise → project
    # ------------------------------------------------------------------
    raw_dosages = extract_subject_dosages(
        subject_vcf, pvar, subject_sample_idx=args.sample_index
    )
    imputed     = impute_missing(raw_dosages, ref_af)
    z           = standardise(imputed, ref_af)
    subject_pcs = project(z, loadings)

    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]
    log.info("Subject PC scores:")
    for i, score in enumerate(subject_pcs[:10]):
        log.info("  PC%-2d = %+.6f", i + 1, score)
    if n_pcs > 10:
        log.info("  … (PC11–PC%d omitted from log)", n_pcs)

    # ------------------------------------------------------------------
    # Step 6: combined CSV
    # ------------------------------------------------------------------
    combined = build_combined_df(eigenvec, panel, subject_pcs, subject_id, n_pcs)
    csv_path = paths["output_dir"] / "subject_pca_scores.csv"
    save_csv(combined, csv_path, n_pcs)

    # Print a compact table to stdout
    subj_row = combined[combined["is_subject"]][pc_cols].iloc[0]
    print(f"\nSubject '{subject_id}' — PC scores")
    print(f"  {'PC':<5}  {'Score':>12}  {'Var%':>8}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*8}")
    var_pct = _variance_explained(eigenval)
    for i in range(n_pcs):
        print(f"  PC{i+1:<3}  {subj_row[f'PC{i+1}']:>+12.6f}  {var_pct[i]:>7.3f}%")

    # ------------------------------------------------------------------
    # Step 7: plots
    # ------------------------------------------------------------------
    pc_pairs: list[tuple[int, int]] = []
    for pair_str in args.pc_pairs.strip().split():
        parts = pair_str.split(",")
        if len(parts) == 2:
            pc_pairs.append((int(parts[0]), int(parts[1])))

    for pc_x, pc_y in pc_pairs:
        if pc_x > n_pcs or pc_y > n_pcs:
            log.warning("PC%d or PC%d exceeds n_pcs=%d — skipping this pair",
                        pc_x, pc_y, n_pcs)
            continue
        out_plot = paths["output_dir"] / f"pca_projection_PC{pc_x}_PC{pc_y}.png"
        plot_pca(combined, pc_x, pc_y, eigenval, subject_id, out_plot)

    scree_path = paths["output_dir"] / "pca_scree.png"
    plot_variance_bar(eigenval, scree_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_found  = int((~np.isnan(raw_dosages)).sum())
    n_imputed = int(np.isnan(raw_dosages).sum())
    cumvar   = sum(var_pct[:10])
    print(f"\nSummary")
    print(f"  Subject VCF       : {subject_vcf}")
    print(f"  Pruned SNPs       : {n_snps:,}")
    print(f"  Genotyped in VCF  : {n_found:,}  ({100*n_found/n_snps:.1f}%)")
    print(f"  Imputed with 2×AF : {n_imputed:,}  ({100*n_imputed/n_snps:.1f}%)")
    print(f"  Variance (PC1–10) : {cumvar:.1f}%")
    print(f"  CSV output        : {csv_path}")
    print(f"  Plots             : {paths['output_dir']}/pca_projection_*.png")
    print(f"  Scree plot        : {scree_path}")


if __name__ == "__main__":
    main()

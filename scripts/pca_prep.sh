#!/usr/bin/env bash
# pca_prep.sh
#
# Prepares a QC-filtered, LD-pruned genotype matrix from the
# 1000 Genomes SAS+EUR reference panel and computes PCA (top 20 PCs).
#
# Steps
# -----
#  1. Convert 1kg_SAS_EUR_merged.vcf.gz to PLINK2 binary format (pgen/pvar/psam).
#  2. QC filters applied in two passes:
#     a. HWE P < 1e-6 computed in EUR samples only → passing SNP list.
#     b. Global filters: genotype missingness > 5%, MAF < 5%,
#        restricted to autosomal SNPs in the HWE-pass list.
#  3. Long-range LD regions excluded by scanning the pvar file with awk:
#     • chr6  25–35 Mb  (HLA)
#     • chr8   7–13 Mb  (8p23.1 inversion)
#  4. LD pruning: sliding window 1000 kb, step 100 variants, r² < 0.1.
#  5. Final pgen built from prune.in list (LRLD SNPs already absent).
#  6. PCA with 20 PCs; eigenvalues and eigenvectors saved.
#  7. Variance-explained table printed to stdout and saved to
#     data/processed/pca/pca_variance_explained.txt.
#
# All intermediate and final outputs go to data/processed/pca/.
#
# Usage
# -----
#   bash scripts/pca_prep.sh [OPTIONS]
#
# Options
#   --threads N        Parallel threads for PLINK2 (default: 4)
#   --plink2  PATH     Path to plink2 binary (default: plink2 on $PATH)
#   --skip-existing    Skip a step if its primary output already exists
#   --dry-run          Print commands without executing them
#
# Requirements
# ------------
#   plink2 >= 2.0  (conda install -c bioconda plink2)
#   awk, sort, wc  (standard Unix tools)
#
# Data dependencies (created by earlier pipeline scripts)
#   data/raw/1kg/1kg_SAS_EUR_merged.vcf.gz   (or path from config.yaml)
#   data/reference/1kg_EUR_samples.txt        (created by download_1kg_sas.sh)

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour / logging helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${BOLD}[pca]${RESET}  $*"; }
success() { echo -e "${GREEN}${BOLD}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[WARN]${RESET}  $*"; }
err()     { echo -e "${RED}${BOLD}[ERR]${RESET}   $*" >&2; }
step()    { echo -e "\n${CYAN}${BOLD}══════  $*  ══════${RESET}"; }

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
THREADS=4
PLINK2_BIN="plink2"
SKIP_EXISTING=false
DRY_RUN=false

usage() {
    sed -n '2,40p' "$0" | grep '^#' | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads)       THREADS="$2";     shift 2 ;;
        --plink2)        PLINK2_BIN="$2";  shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --dry-run)       DRY_RUN=true;     shift ;;
        --help|-h)       usage ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# Wrapper: prints and optionally runs a command
run() {
    echo -e "  ${BOLD}\$${RESET} $*"
    if [[ "$DRY_RUN" == false ]]; then
        "$@"
    fi
}

# Skip step if primary output exists and --skip-existing is set
skip_if_exists() {
    local output="$1"
    if [[ "$SKIP_EXISTING" == true && -e "$output" ]]; then
        warn "Output exists, skipping: $(basename "$output")  (--skip-existing)"
        return 0   # signal: skip this step
    fi
    return 1       # signal: proceed
}

# ---------------------------------------------------------------------------
# Paths — all relative to project root (one level up from scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

INPUT_VCF="$ROOT/data/raw/1kg/1kg_SAS_EUR_merged.vcf.gz"
EUR_SAMPLES="$ROOT/data/reference/1kg_EUR_samples.txt"
OUTDIR="$ROOT/data/processed/pca"

# Intermediate prefixes (all under $OUTDIR)
RAW="$OUTDIR/step1_raw"
HWE_EUR="$OUTDIR/step2a_hwe_eur"
QC="$OUTDIR/step2b_qc"
LRLD_LIST="$OUTDIR/step3_lrld_exclude.snplist"
PRUNE="$OUTDIR/step4_prune"
PRUNED="$OUTDIR/step5_pruned"
PCA="$OUTDIR/step6_pca"

mkdir -p "$OUTDIR"

# ---------------------------------------------------------------------------
# QC thresholds (edit here to change pipeline-wide)
# ---------------------------------------------------------------------------
GENO_THRESH=0.05     # drop SNPs with genotype call rate < 95 %
MAF_THRESH=0.05      # drop SNPs with MAF < 5 %
HWE_P=1e-6           # HWE P-value threshold (EUR-only)
N_PCS=20             # number of principal components

# Long-range LD exclusion coordinates (hg19/GRCh37, 1-based bp)
HLA_CHR=6;  HLA_START=25000000;  HLA_END=35000000    # MHC / HLA
INV_CHR=8;  INV_START=7000000;   INV_END=13000000    # 8p23.1 inversion

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
step "Checking prerequisites"

if ! command -v "$PLINK2_BIN" &>/dev/null; then
    if [[ "$DRY_RUN" == true ]]; then
        warn "plink2 not found — dry-run will print commands only."
        PLINK2_VER="(not installed)"
    else
        err "plink2 not found at '${PLINK2_BIN}'."
        err "Install via: conda install -c bioconda plink2"
        err "Or pass --plink2 /path/to/plink2"
        exit 1
    fi
else
    PLINK2_VER=$("$PLINK2_BIN" --version 2>&1 | head -1)
fi
info "plink2     : $PLINK2_VER"
info "Threads    : $THREADS"
info "Input VCF  : $INPUT_VCF"
info "EUR samples: $EUR_SAMPLES"
info "Output dir : $OUTDIR"

if [[ ! -f "$INPUT_VCF" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        warn "Input VCF not found (dry-run): $INPUT_VCF"
    else
        err "Input VCF not found: $INPUT_VCF"
        err "Run scripts/download_1kg_sas.sh first."
        exit 1
    fi
fi

if [[ ! -f "$EUR_SAMPLES" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        warn "EUR sample list not found (dry-run): $EUR_SAMPLES"
    else
        err "EUR sample list not found: $EUR_SAMPLES"
        err "Run scripts/download_1kg_sas.sh first (it creates this file)."
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# Step 1 — VCF → PLINK2 binary (pgen / pvar / psam)
# ---------------------------------------------------------------------------
step "Step 1: VCF → PLINK2 binary"

if ! skip_if_exists "${RAW}.pgen"; then
    info "Converting VCF to pgen format …"
    run "$PLINK2_BIN" \
        --vcf "$INPUT_VCF" \
        --double-id \
        --chr 1-22 \
        --max-alleles 2 \
        --vcf-min-gp 0 \
        --make-pgen \
        --threads "$THREADS" \
        --out "$RAW"

    if [[ "$DRY_RUN" == false ]]; then
        N_VAR=$(wc -l < "${RAW}.pvar")
        N_SAM=$(awk 'NR > 1' "${RAW}.psam" | wc -l)
        success "Raw pgen: $((N_VAR - 1)) variants, $N_SAM samples"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2a — HWE filter in EUR samples only
# ---------------------------------------------------------------------------
step "Step 2a: EUR-only HWE filter (P < ${HWE_P})"

# PLINK2 needs a two-column keep file: FID  IID
# With --double-id above, FID = IID = sample_id
EUR_KEEP="$OUTDIR/eur_keep.txt"
if [[ "$DRY_RUN" == false && ! -f "$EUR_KEEP" ]]; then
    awk '{print $1, $1}' "$EUR_SAMPLES" > "$EUR_KEEP"
    info "EUR keep file: $(wc -l < "$EUR_KEEP") samples → $EUR_KEEP"
elif [[ "$DRY_RUN" == true ]]; then
    info "EUR keep file (dry-run): $EUR_KEEP  (awk '{print \$1,\$1}' $EUR_SAMPLES)"
fi

if ! skip_if_exists "${HWE_EUR}.snplist"; then
    info "Computing HWE in EUR subset and writing passing SNP list …"
    run "$PLINK2_BIN" \
        --pfile "$RAW" \
        --keep "$EUR_KEEP" \
        --hwe "$HWE_P" midp \
        --write-snplist \
        --no-psam-pheno \
        --threads "$THREADS" \
        --out "$HWE_EUR"

    if [[ "$DRY_RUN" == false ]]; then
        N_HWE=$(wc -l < "${HWE_EUR}.snplist")
        success "SNPs passing HWE in EUR: $N_HWE"
    fi
fi

# ---------------------------------------------------------------------------
# Step 2b — Global QC: missingness, MAF, restrict to HWE-pass list
# ---------------------------------------------------------------------------
step "Step 2b: Global QC (geno < ${GENO_THRESH}, MAF > ${MAF_THRESH})"

if ! skip_if_exists "${QC}.pgen"; then
    info "Applying missingness, MAF filters to full sample set …"
    run "$PLINK2_BIN" \
        --pfile "$RAW" \
        --extract "${HWE_EUR}.snplist" \
        --geno "$GENO_THRESH" \
        --maf  "$MAF_THRESH" \
        --make-pgen \
        --threads "$THREADS" \
        --out "$QC"

    if [[ "$DRY_RUN" == false ]]; then
        N_QC=$(awk 'NR > 1' "${QC}.pvar" | wc -l)
        success "Post-QC variants: $N_QC"
    fi
fi

# ---------------------------------------------------------------------------
# Step 3 — Long-range LD region exclusion (awk scan of pvar)
# ---------------------------------------------------------------------------
step "Step 3: Long-range LD region exclusion"

info "Regions excluded:"
info "  chr${HLA_CHR}  ${HLA_START}–${HLA_END} bp  (HLA / MHC)"
info "  chr${INV_CHR}  ${INV_START}–${INV_END} bp  (8p23.1 inversion)"

if ! skip_if_exists "$LRLD_LIST"; then
    if [[ "$DRY_RUN" == true ]]; then
        info "(dry-run) Would scan ${QC}.pvar with awk to extract LRLD variant IDs → $LRLD_LIST"
    else
        # pvar columns: #CHROM  POS  ID  REF  ALT
        # Strip any "chr" prefix so matching works for both chr6 and 6 conventions
        awk -v hla_chr="$HLA_CHR" -v hla_s="$HLA_START" -v hla_e="$HLA_END" \
            -v inv_chr="$INV_CHR" -v inv_s="$INV_START" -v inv_e="$INV_END" \
        '!/^#/ {
            chr = $1; sub(/^chr/, "", chr)
            pos = $2 + 0
            if ((chr == hla_chr && pos >= hla_s && pos <= hla_e) ||
                (chr == inv_chr && pos >= inv_s && pos <= inv_e))
                print $3
        }' "${QC}.pvar" | sort -u > "$LRLD_LIST"
        N_LRLD=$(wc -l < "$LRLD_LIST")
        success "LRLD variants to exclude: $N_LRLD  → $LRLD_LIST"
    fi
fi

# ---------------------------------------------------------------------------
# Step 4 — LD pruning (on LRLD-excluded set)
# ---------------------------------------------------------------------------
step "Step 4: LD pruning (window 1000 kb, step 100, r² < 0.1)"

if ! skip_if_exists "${PRUNE}.prune.in"; then
    info "Running --indep-pairwise to generate prune.in / prune.out …"
    run "$PLINK2_BIN" \
        --pfile "$QC" \
        --exclude "$LRLD_LIST" \
        --indep-pairwise 1000kb 100 0.1 \
        --threads "$THREADS" \
        --out "$PRUNE"

    if [[ "$DRY_RUN" == false ]]; then
        N_IN=$(wc -l  < "${PRUNE}.prune.in")
        N_OUT=$(wc -l < "${PRUNE}.prune.out")
        success "Pruning complete: $N_IN retained, $N_OUT removed"
    fi
fi

# ---------------------------------------------------------------------------
# Step 5 — Build final pruned pgen
# ---------------------------------------------------------------------------
step "Step 5: Applying prune list → final pgen"

if ! skip_if_exists "${PRUNED}.pgen"; then
    info "Extracting prune.in SNPs (LRLD exclusion already reflected) …"
    run "$PLINK2_BIN" \
        --pfile "$QC" \
        --extract "${PRUNE}.prune.in" \
        --exclude "$LRLD_LIST" \
        --make-pgen \
        --threads "$THREADS" \
        --out "$PRUNED"

    if [[ "$DRY_RUN" == false ]]; then
        N_FINAL=$(awk 'NR > 1' "${PRUNED}.pvar" | wc -l)
        N_SAM=$(awk 'NR > 1' "${PRUNED}.psam" | wc -l)
        success "Final pruned pgen: $N_FINAL variants, $N_SAM samples"
    fi
fi

# ---------------------------------------------------------------------------
# Step 6 — PCA: top 20 principal components
# ---------------------------------------------------------------------------
step "Step 6: PCA (top ${N_PCS} PCs)"

if ! skip_if_exists "${PCA}.eigenval"; then
    info "Running PCA with variant loadings (var-wts) …"
    run "$PLINK2_BIN" \
        --pfile "$PRUNED" \
        --pca "$N_PCS" approx var-wts \
        --threads "$THREADS" \
        --out "$PCA"

    success "Eigenvalues      : ${PCA}.eigenval"
    success "Eigenvectors     : ${PCA}.eigenvec"
    success "Variant loadings : ${PCA}.eigenvec.var  (required by project_subject_pca.py)"
fi

# ---------------------------------------------------------------------------
# Step 6b — Allele frequencies (required for subject projection / imputation)
# ---------------------------------------------------------------------------
step "Step 6b: Allele frequencies for imputation"

if ! skip_if_exists "${PCA}.afreq"; then
    info "Computing ALT allele frequencies on the pruned pgen …"
    run "$PLINK2_BIN" \
        --pfile "$PRUNED" \
        --freq \
        --threads "$THREADS" \
        --out "$PCA"

    success "Allele frequencies: ${PCA}.afreq  (required by project_subject_pca.py)"
fi

# ---------------------------------------------------------------------------
# Step 7 — Variance explained per PC
# ---------------------------------------------------------------------------
step "Step 7: Variance explained per PC"

VAREXP_FILE="$OUTDIR/pca_variance_explained.txt"

if [[ ! -f "${PCA}.eigenval" ]]; then
    if [[ "$DRY_RUN" == true ]]; then
        warn "Eigenvalue file not yet present (dry-run) — variance table skipped."
        exit 0
    fi
    err "Eigenvalue file not found: ${PCA}.eigenval"
    err "PCA step may have failed. Check PLINK2 log: ${PCA}.log"
    exit 1
fi

# Compute proportion and cumulative variance from eigenvalues.
# PLINK2 eigenvalues are already variance values (λ), not std devs.
awk -v n_pcs="$N_PCS" '
BEGIN {
    sum = 0
}
{
    ev[NR] = $1
    sum   += $1
}
END {
    cum = 0
    printf "%-6s  %12s  %10s  %12s\n", "PC", "Eigenvalue", "Var %", "Cumul %"
    printf "%-6s  %12s  %10s  %12s\n", "------", "------------", "----------", "------------"
    for (i = 1; i <= NR; i++) {
        pct  = ev[i] / sum * 100
        cum += pct
        printf "%-6d  %12.4f  %9.4f%%  %11.4f%%\n", i, ev[i], pct, cum
    }
    printf "\nTotal variance explained by top %d PCs: %.2f%%\n", NR, cum
}
' "${PCA}.eigenval" | tee "$VAREXP_FILE"

success "Variance table saved → $VAREXP_FILE"

# ---------------------------------------------------------------------------
# Summary manifest
# ---------------------------------------------------------------------------
MANIFEST="$OUTDIR/pca_prep_manifest.txt"
{
    echo "# PCA preparation manifest"
    echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo ""
    echo "input_vcf:            $INPUT_VCF"
    echo "output_dir:           $OUTDIR"
    echo ""
    echo "qc_geno_threshold:    $GENO_THRESH"
    echo "qc_maf_threshold:     $MAF_THRESH"
    echo "qc_hwe_p_threshold:   $HWE_P  (EUR samples only)"
    echo "lrld_hla:             chr${HLA_CHR}:${HLA_START}-${HLA_END}"
    echo "lrld_chr8_inversion:  chr${INV_CHR}:${INV_START}-${INV_END}"
    echo "ld_window_kb:         1000"
    echo "ld_step_variants:     100"
    echo "ld_r2_threshold:      0.1"
    echo "n_pcs:                $N_PCS"
    echo ""
    echo "# File registry"
    echo "step1_raw_pgen:       ${RAW}.pgen"
    echo "step2_qc_pgen:        ${QC}.pgen"
    echo "step3_lrld_exclude:   ${LRLD_LIST}"
    echo "step4_prune_in:       ${PRUNE}.prune.in"
    echo "step4_prune_out:      ${PRUNE}.prune.out"
    echo "step5_pruned_pgen:    ${PRUNED}.pgen"
    echo "step6_eigenvalues:    ${PCA}.eigenval"
    echo "step6_eigenvectors:   ${PCA}.eigenvec"
    echo "step6_var_loadings:   ${PCA}.eigenvec.var"
    echo "step6b_afreq:         ${PCA}.afreq"
    echo "step7_variance_table: ${VAREXP_FILE}"
} > "$MANIFEST"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════${RESET}"
echo -e "${GREEN}${BOLD}  PCA preparation complete.${RESET}"
echo -e "${GREEN}${BOLD}  Eigenvectors : ${PCA}.eigenvec${RESET}"
echo -e "${GREEN}${BOLD}  Eigenvalues  : ${PCA}.eigenval${RESET}"
echo -e "${GREEN}${BOLD}  Variance tbl : ${VAREXP_FILE}${RESET}"
echo -e "${GREEN}${BOLD}  Manifest     : ${MANIFEST}${RESET}"
echo -e "${GREEN}${BOLD}════════════════════════════════════════════════════${RESET}"

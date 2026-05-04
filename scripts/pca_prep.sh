#!/usr/bin/env bash
# ==========================================================================
#  pca_prep.sh — Prepare 1000 Genomes SAS+EUR data for PCA (PLINK 1.9)
# ==========================================================================

set -e
set -u
set -o pipefail

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR="data/processed"
PCA_DIR="${DATA_DIR}/pca"
MERGED_VCF="${DATA_DIR}/1kg_SAS_EUR_merged.vcf.gz"
PLINK_RAW="${PCA_DIR}/1kg_sas_eur_raw"
PLINK_QC="${PCA_DIR}/1kg_sas_eur_qc"
PLINK_PRUNED="${PCA_DIR}/1kg_sas_eur_pruned"
PLINK_FINAL="${PCA_DIR}/1kg_sas_eur_final"
PCA_OUTPUT="${PCA_DIR}/1kg_sas_eur_pca"
EXCLUDE_REGIONS="${PCA_DIR}/long_range_ld_regions.txt"
LD_RANGE_SNPS="${PCA_DIR}/ld_range_snps.snplist"

# ── Colour helpers ─────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BOLD}[INFO]${NC}  $*"; }
ok()   { echo -e "${GREEN}[  OK]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC}  $*"; }

mkdir -p "${PCA_DIR}"

# ── Step 1: Convert VCF to PLINK 1.9 binary ───────────────────────────────
info "━━━ Step 1: Convert VCF to PLINK 1.9 binary format ━━━"
if [ -f "${PLINK_RAW}.bed" ]; then
    info "PLINK binary already exists — skipping conversion."
else
    # PLINK 1.9 version
    # Note: --set-missing-var-ids '@:#' ensures we have unique IDs (CHR:POS)
    plink \
        --vcf "${MERGED_VCF}" \
        --make-bed \
        --set-missing-var-ids '@:#' \
        --out "${PLINK_RAW}" \
        --allow-extra-chr \
        --threads 4
    ok "Conversion complete."
fi

# ── Step 2: Apply QC filters ──────────────────────────────────────────────
info "━━━ Step 2: Apply QC filters ━━━"
if [ -f "${PLINK_QC}.bed" ]; then
    info "QC'd fileset already exists — skipping."
else
    # Filters: MAF > 1%, Geno < 5%, HWE > 1e-6
    plink \
        --bfile "${PLINK_RAW}" \
        --maf 0.01 \
        --geno 0.05 \
        --hwe 1e-6 \
        --make-bed \
        --out "${PLINK_QC}" \
        --allow-extra-chr
    ok "QC filters applied."
fi

# ── Step 3: LD Pruning ────────────────────────────────────────────────────
info "━━━ Step 3: LD Pruning ━━━"
if [ -f "${PCA_DIR}/ld_prune.prune.in" ]; then
    info "LD pruning already performed — skipping."
else
    # Indep-pairwise: window size 50kb, step 5, r^2 threshold 0.2
    plink \
        --bfile "${PLINK_QC}" \
        --indep-pairwise 50 5 0.2 \
        --out "${PCA_DIR}/ld_prune"
    ok "LD pruning complete."
fi

# ── Step 4: Exclude long-range LD regions ─────────────────────────────────
info "━━━ Step 4: Exclude long-range LD regions ━━━"

# Create regions file (CHR START END LABEL)
cat > "${EXCLUDE_REGIONS}" << 'EOF'
6 25000000 35000000 HLA_MHC
8 7000000 13000000 chr8_inversion
11 46000000 57000000 chr11_inversion
17 40000000 45000000 chr17_inversion
EOF

if [ -f "${PLINK_FINAL}.bed" ]; then
    info "Final PCA-ready fileset already exists — skipping."
else
    # Step A: Identify SNPs in long-range LD regions (PLINK 1.9 workaround for --exclude-range)
    plink \
        --bfile "${PLINK_QC}" \
        --extract range "${EXCLUDE_REGIONS}" \
        --write-snplist \
        --out "${PCA_DIR}/ld_range_snps"

    # Step B: Create final fileset (Pruned SNPs minus LD regions)
    plink \
        --bfile "${PLINK_QC}" \
        --extract "${PCA_DIR}/ld_prune.prune.in" \
        --exclude "${PCA_DIR}/ld_range_snps.snplist" \
        --make-bed \
        --out "${PLINK_FINAL}"
    ok "Long-range LD exclusion complete."
fi

# ── Step 5: Run PCA ───────────────────────────────────────────────────────
info "━━━ Step 5: Compute PCA (top 20 PCs) ━━━"
if [ -f "${PCA_OUTPUT}.eigenvec" ]; then
    info "PCA results already exist — skipping."
else
    plink \
        --bfile "${PLINK_FINAL}" \
        --pca 20 \
        --out "${PCA_OUTPUT}"
    ok "PCA computation complete."
fi

info "Pipeline finished successfully."

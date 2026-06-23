#!/usr/bin/env bash
# ===========================================================================
#  ld_clump.sh — Run LD clumping on cleaned GWAS summary statistics
#  using 1000 Genomes South Asian (SAS) cohort as reference.
#
#  Parameters: r2=0.1, window=1000kb, p1=5e-8
# ===========================================================================

set -e
set -u
set -o pipefail

# ── Configuration ──────────────────────────────────────────────────────────
CLEANED_DIR="data/processed/gwas_cleaned"
CLUMPED_DIR="data/processed/gwas_clumped"
REF_BFILE="data/processed/pca/1kg_sas_eur_qc"
SAS_SAMPLES="data/reference/1kg_SAS_samples.txt"
SAS_KEEP="data/processed/sas_keep.txt"

# ── Colour helpers ─────────────────────────────────────────────────────────
BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BOLD}[INFO]${NC}  $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC}  $*"; }
ok()   { echo -e "${GREEN}[  OK]${NC}  $*"; }
fail() { echo -e "${RED}[FAIL]${NC}  $*"; }

mkdir -p "${CLUMPED_DIR}"

# ── Check inputs ───────────────────────────────────────────────────────────
if [ ! -f "${SAS_SAMPLES}" ]; then
    fail "SAS samples file not found at '${SAS_SAMPLES}'"
    exit 1
fi

if [ ! -f "${REF_BFILE}.bed" ]; then
    fail "Reference panel binaries not found at '${REF_BFILE}.bed'"
    exit 1
fi

# ── Create 2-column keep file ──────────────────────────────────────────────
info "Creating 2-column SAS keep file at '${SAS_KEEP}'..."
awk '{print $1, $1}' "${SAS_SAMPLES}" > "${SAS_KEEP}"
ok "Keep file created with $(wc -l < "${SAS_KEEP}" | xargs) samples."

# ── Detect PLINK command ───────────────────────────────────────────────────
PLINK_CMD=""
PLINK_VER=""

if command -v plink2 &>/dev/null; then
    PLINK_CMD="plink2"
    PLINK_VER="2"
    info "Using PLINK 2.0 executable on PATH."
elif conda run -n dna_facial plink2 --version &>/dev/null; then
    PLINK_CMD="conda run -n dna_facial plink2"
    PLINK_VER="2"
    info "Using PLINK 2.0 inside conda environment 'dna_facial'."
elif command -v plink &>/dev/null; then
    PLINK_CMD="plink"
    PLINK_VER="1.9"
    warn "plink2 not found on PATH. Falling back to PLINK 1.9 executable."
elif conda run -n dna_facial plink --version &>/dev/null; then
    PLINK_CMD="conda run -n dna_facial plink"
    PLINK_VER="1.9"
    warn "plink2 not found. Falling back to PLINK 1.9 inside conda environment 'dna_facial'."
else
    fail "Neither plink2 nor plink (1.9) could be located. Please check your PATH or environment."
    exit 1
fi

# ── Loop over cleaned GWAS files ───────────────────────────────────────────
cleaned_files=(${CLEANED_DIR}/*_cleaned.txt)

if [ ${#cleaned_files[@]} -eq 0 ] || [ ! -e "${cleaned_files[0]}" ]; then
    fail "No cleaned GWAS files found in '${CLEANED_DIR}'. Please run clean_gwas.py first."
    exit 1
fi

info "Found ${#cleaned_files[@]} cleaned GWAS files."

for gwas_file in "${cleaned_files[@]}"; do
    filename=$(basename "${gwas_file}")
    trait="${filename%_cleaned.txt}"
    
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    info "Running LD clumping for trait: ${trait}"
    info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    out_prefix="${CLUMPED_DIR}/${trait}"
    
    # Run clumping based on PLINK version
    if [ "${PLINK_VER}" = "2" ]; then
        # PLINK 2.0 Clumping
        # --clump-field P and --clump-snp-field SNP match standardised output columns
        ${PLINK_CMD} \
            --bfile "${REF_BFILE}" \
            --keep "${SAS_KEEP}" \
            --clump "${gwas_file}" \
            --clump-field P \
            --clump-snp-field SNP \
            --clump-r2 0.1 \
            --clump-kb 1000 \
            --clump-p1 5e-8 \
            --out "${out_prefix}" \
            --allow-extra-chr
            
        # PLINK 2.0 outputs to .clumps (or .clumps.zst)
        clump_file="${out_prefix}.clumps"
        if [ -f "${clump_file}.zst" ]; then
            info "Decompressing PLINK2 zstd clumps output..."
            zstd -d --rm "${clump_file}.zst"
        fi
    else
        # PLINK 1.9 Clumping
        ${PLINK_CMD} \
            --bfile "${REF_BFILE}" \
            --keep "${SAS_KEEP}" \
            --clump "${gwas_file}" \
            --clump-field P \
            --clump-snp-field SNP \
            --clump-r2 0.1 \
            --clump-kb 1000 \
            --clump-p1 5e-8 \
            --out "${out_prefix}" \
            --allow-extra-chr
            
        # PLINK 1.9 outputs to .clumped
        clump_file="${out_prefix}.clumped"
    fi
    
    if [ ! -f "${clump_file}" ]; then
        warn "Clumping file '${clump_file}' was not generated. This traits might have no significant clumped variants."
        # Create empty lead SNP file
        echo -n "" > "${out_prefix}_lead_snps.txt"
        continue
    fi
    
    # ── Extract lead SNPs using Python ─────────────────────────────────────
    lead_snps_file="${out_prefix}_lead_snps.txt"
    info "Extracting lead SNPs from '${clump_file}' to '${lead_snps_file}'..."
    
    python3 -c "
import sys, os
in_path = sys.argv[1]
out_path = sys.argv[2]
snps = []
if os.path.exists(in_path):
    with open(in_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            # Skip headers or empty lines
            if parts[0].upper() in ['CHR', '#CHR', 'INDEX'] or parts[0].startswith('('):
                continue
            # Column indices: PLINK 1.9/PLINK2 has SNP ID at index 2 (third column)
            if len(parts) >= 3:
                snps.append(parts[2])
with open(out_path, 'w') as out:
    out.write('\n'.join(snps) + '\n')
" "${clump_file}" "${lead_snps_file}"
    
    n_lead=$(wc -l < "${lead_snps_file}" | xargs)
    ok "LD clumping complete. Found ${n_lead} independent lead SNPs."
done

info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ok "All clumping runs finished. Results saved in '${CLUMPED_DIR}'."

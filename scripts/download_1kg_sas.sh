#!/usr/bin/env bash
# download_1kg_sas.sh
#
# Downloads 1000 Genomes Phase 3 VCFs, subsets to SAS+EUR samples, applies
# biallelic SNP / MAF / missingness filters, and merges all chromosomes into
# one indexed VCF.
#
# Idempotent: already-downloaded files and already-filtered per-chromosome
# VCFs are reused; only missing pieces are fetched or reprocessed.
#
# Usage:
#   bash scripts/download_1kg_sas.sh [--threads N] [--keep-raw]
#
# Options:
#   --threads N    bcftools thread count (default: 4)
#   --keep-raw     do not delete per-chromosome raw downloads after filtering
#
# Requirements: bcftools >= 1.17, tabix, wget or curl (auto-detected)
# All tools are installed by setup_env.sh via conda-forge / bioconda.

set -euo pipefail

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BOLD='\033[1m';   RESET='\033[0m'

info()    { echo -e "${BOLD}[1kg]${RESET}  $*"; }
success() { echo -e "${GREEN}${BOLD}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}${BOLD}[WARN]${RESET}  $*"; }
err()     { echo -e "${RED}${BOLD}[ERR]${RESET}   $*" >&2; }

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
THREADS=4
KEEP_RAW=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --threads) THREADS="$2"; shift 2 ;;
        --keep-raw) KEEP_RAW=true; shift ;;
        *) err "Unknown argument: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Paths — all relative to project root (one level up from scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"

RAW_DIR="$ROOT/data/raw/1kg"
REF_DIR="$ROOT/data/reference"
PROCESSED_DIR="$ROOT/data/processed"
FILTERED_DIR="$RAW_DIR/filtered"   # per-chromosome filtered VCFs (intermediate)

SAS_SAMPLES="$REF_DIR/1kg_SAS_samples.txt"
EUR_SAMPLES="$REF_DIR/1kg_EUR_samples.txt"
MERGED_VCF="$PROCESSED_DIR/1kg_SAS_EUR_merged.vcf.gz"

mkdir -p "$RAW_DIR" "$REF_DIR" "$PROCESSED_DIR" "$FILTERED_DIR"

# ---------------------------------------------------------------------------
# 1000 Genomes FTP base (EBI mirror — more reliable than NCBI for large files)
# ---------------------------------------------------------------------------
BASE_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502"
PANEL_URL="${BASE_URL}/integrated_call_samples_v3.20130502.ALL.panel"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------
info "Checking prerequisites..."

for tool in bcftools tabix; do
    if ! command -v "$tool" &>/dev/null; then
        err "$tool not found. Run bash setup_env.sh first."
        exit 1
    fi
done

# Prefer wget; fall back to curl
if command -v wget &>/dev/null; then
    DOWNLOADER="wget"
elif command -v curl &>/dev/null; then
    DOWNLOADER="curl"
else
    err "Neither wget nor curl found. Install one and retry."
    exit 1
fi
info "Downloader: $DOWNLOADER  |  bcftools: $(bcftools --version | head -1)  |  threads: $THREADS"

# ---------------------------------------------------------------------------
# Download helper — resumes interrupted transfers
# ---------------------------------------------------------------------------
download() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        info "Already exists, skipping: $(basename "$dest")"
        return 0
    fi
    info "Downloading: $(basename "$dest")"
    if [[ "$DOWNLOADER" == "wget" ]]; then
        wget --continue --no-verbose --show-progress -O "${dest}.part" "$url"
    else
        curl -L --continue-at - --progress-bar -o "${dest}.part" "$url"
    fi
    mv "${dest}.part" "$dest"
    success "Downloaded: $(basename "$dest")"
}

# Same as download() but also fetches the .tbi index alongside the VCF
download_with_index() {
    local url="$1"
    local dest="$2"
    download "$url"       "$dest"
    download "${url}.tbi" "${dest}.tbi"
}

# ---------------------------------------------------------------------------
# Step 1 — Panel file: extract SAS and EUR sample IDs
# ---------------------------------------------------------------------------
PANEL_FILE="$RAW_DIR/integrated_call_samples_v3.panel"

info "--- Step 1: Sample panel ---"
download "$PANEL_URL" "$PANEL_FILE"

# Column layout: sample  pop  super_pop  gender
if [[ ! -s "$SAS_SAMPLES" ]]; then
    awk '$3 == "SAS" {print $1}' "$PANEL_FILE" > "$SAS_SAMPLES"
    SAS_N=$(wc -l < "$SAS_SAMPLES")
    success "SAS samples: $SAS_N  ->  $SAS_SAMPLES"
else
    SAS_N=$(wc -l < "$SAS_SAMPLES")
    warn "SAS sample list already exists ($SAS_N samples). Skipping extraction."
fi

if [[ ! -s "$EUR_SAMPLES" ]]; then
    awk '$3 == "EUR" {print $1}' "$PANEL_FILE" > "$EUR_SAMPLES"
    EUR_N=$(wc -l < "$EUR_SAMPLES")
    success "EUR samples: $EUR_N  ->  $EUR_SAMPLES"
else
    EUR_N=$(wc -l < "$EUR_SAMPLES")
    warn "EUR sample list already exists ($EUR_N samples). Skipping extraction."
fi

# Combined sample list used by bcftools view -S
COMBINED_SAMPLES="$RAW_DIR/1kg_SAS_EUR_combined.txt"
cat "$SAS_SAMPLES" "$EUR_SAMPLES" > "$COMBINED_SAMPLES"
COMBINED_N=$(wc -l < "$COMBINED_SAMPLES")
info "Combined SAS+EUR sample list: $COMBINED_N samples"

# ---------------------------------------------------------------------------
# Step 2 — Per-chromosome download + filter
# ---------------------------------------------------------------------------
info "--- Step 2: Per-chromosome download and filtering ---"

FILTERED_VCFS=()   # accumulates paths for the merge step

for CHR in $(seq 1 22); do
    VCF_FILENAME="ALL.chr${CHR}.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz"
    RAW_VCF="$RAW_DIR/$VCF_FILENAME"
    FILT_VCF="$FILTERED_DIR/chr${CHR}.SAS_EUR.filtered.vcf.gz"

    FILTERED_VCFS+=("$FILT_VCF")

    # -- Download raw chromosome VCF (skip if present) ---------------------
    RAW_URL="${BASE_URL}/${VCF_FILENAME}"
    download_with_index "$RAW_URL" "$RAW_VCF"

    # -- Filter (skip if filtered output already exists) -------------------
    if [[ -f "$FILT_VCF" && -f "${FILT_VCF}.tbi" ]]; then
        warn "chr${CHR}: filtered VCF already exists. Skipping."
        continue
    fi

    info "chr${CHR}: filtering..."

    # bcftools pipeline (single pass, piped):
    #   view  -S   subset to SAS+EUR samples
    #   view  -m2 -M2 -v snps   keep only biallelic SNPs
    #   view  -q 0.01:minor     MAF > 0.01 in the retained samples
    #   view  -e 'F_MISSING>0.05'  drop SNPs with >5% missingness
    #   annotate --set-id   give each SNP a stable chr:pos:ref:alt ID
    #   output: bgzf-compressed
    bcftools view \
        --samples-file "$COMBINED_SAMPLES" \
        --force-samples \
        --threads "$THREADS" \
        "$RAW_VCF" \
    | bcftools view \
        --min-alleles 2 --max-alleles 2 \
        --type snps \
        --threads "$THREADS" \
    | bcftools view \
        --min-af "0.01:minor" \
        --threads "$THREADS" \
    | bcftools view \
        --exclude 'F_MISSING > 0.05' \
        --threads "$THREADS" \
    | bcftools annotate \
        --set-id '%CHROM:%POS:%REF:%ALT' \
        --threads "$THREADS" \
        --output-type z \
        --output "$FILT_VCF"

    tabix -p vcf "$FILT_VCF"

    SNPS=$(bcftools stats "$FILT_VCF" | awk '/^SN.*number of SNPs/{print $NF}')
    success "chr${CHR}: $SNPS SNPs retained -> $(basename "$FILT_VCF")"

    # Remove raw chromosome VCF to reclaim disk space (unless --keep-raw)
    if [[ "$KEEP_RAW" == false ]]; then
        rm -f "$RAW_VCF" "${RAW_VCF}.tbi"
        info "chr${CHR}: raw VCF removed (use --keep-raw to retain)."
    fi
done

# ---------------------------------------------------------------------------
# Step 3 — Merge all chromosomes
# ---------------------------------------------------------------------------
info "--- Step 3: Merging chromosomes 1-22 ---"

if [[ -f "$MERGED_VCF" && -f "${MERGED_VCF}.tbi" ]]; then
    warn "Merged VCF already exists: $MERGED_VCF"
    warn "Delete it and re-run if you want to rebuild from filtered per-chr VCFs."
else
    # Write a sorted file list for bcftools concat
    FILT_LIST="$FILTERED_DIR/filtered_vcf_list.txt"
    printf '%s\n' "${FILTERED_VCFS[@]}" > "$FILT_LIST"

    # Verify every filtered VCF exists before attempting the merge
    MISSING_ANY=false
    for f in "${FILTERED_VCFS[@]}"; do
        if [[ ! -f "$f" ]]; then
            err "Missing filtered VCF: $f"
            MISSING_ANY=true
        fi
    done
    if [[ "$MISSING_ANY" == true ]]; then
        err "One or more filtered VCFs are missing. Cannot merge. Exiting."
        exit 1
    fi

    info "Concatenating 22 filtered VCFs..."
    bcftools concat \
        --file-list "$FILT_LIST" \
        --allow-overlaps \
        --threads "$THREADS" \
        --output-type z \
        --output "$MERGED_VCF"

    tabix -p vcf "$MERGED_VCF"
    success "Merged VCF: $MERGED_VCF"

    # Summary stats on the merged file
    TOTAL_SNPS=$(bcftools stats "$MERGED_VCF" | awk '/^SN.*number of SNPs/{print $NF}')
    TOTAL_SAMPLES=$(bcftools query -l "$MERGED_VCF" | wc -l)
    success "Total SNPs in merged panel : $TOTAL_SNPS"
    success "Total samples in merged panel: $TOTAL_SAMPLES"

    # Persist a small manifest alongside the merged VCF
    MANIFEST="$PROCESSED_DIR/1kg_SAS_EUR_manifest.txt"
    {
        echo "# 1000 Genomes Phase 3 SAS+EUR reference panel"
        echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "merged_vcf: $MERGED_VCF"
        echo "total_snps: $TOTAL_SNPS"
        echo "total_samples: $TOTAL_SAMPLES"
        echo "sas_samples: $(wc -l < "$SAS_SAMPLES")"
        echo "eur_samples: $(wc -l < "$EUR_SAMPLES")"
        echo "filters_applied: biallelic_snps MAF>0.01 missingness<0.05"
        echo "source_url: $BASE_URL"
    } > "$MANIFEST"
    success "Manifest written: $MANIFEST"
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo -e "${GREEN}${BOLD}=================================================${RESET}"
echo -e "${GREEN}${BOLD}  1KG SAS+EUR reference panel ready.            ${RESET}"
echo -e "${GREEN}${BOLD}  $MERGED_VCF${RESET}"
echo -e "${GREEN}${BOLD}=================================================${RESET}"

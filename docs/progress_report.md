# Project Progress Report
## DNA-Based Facial Approximation for Indian Forensic Contexts

**Submitted to:** Project Guide
**Date:** April 2026
**Status:** Project setup and data acquisition phase complete

---

## 1. Project Overview

This project develops a computational pipeline that takes a **VCF file of SNP genotypes** as input and produces an **annotated 2D facial composite** calibrated for Indian forensic contexts. The key motivation is that all existing DNA phenotyping tools are built on European cohort data and are not directly applicable to Indian populations. This pipeline incorporates South Asian reference panels, Indian-specific allele frequencies, and GWAS loci replicated in South Asian samples.

The pipeline is intended for eventual publication in **IEEE Access** and all methodology decisions are being recorded in a living methods document (`docs/methods.md`).

---

## 2. Pipeline Architecture

The full pipeline has five stages, all documented and scaffolded:

```
VCF (SNP genotypes)
  │
  ├─ Stage 1.  Quality control & ancestry inference
  │             └─ PCA projection against 1000 Genomes SAS reference
  │             └─ Random Forest soft classification → ancestry proportions
  │
  ├─ Stage 2.  Pigmentation prediction
  │             └─ HIrisPlex-S (41 SNPs) → eye colour, hair colour, skin colour
  │
  ├─ Stage 3.  Facial morphology estimation
  │             └─ Ancestry-weighted polygenic risk scores from GWAS loci
  │             └─ Landmark displacement model
  │
  ├─ Stage 4.  Composite generation
  │             └─ Delaunay triangulation / thin-plate spline warping
  │             └─ Pigmentation overlay (RGB/HSV)
  │
  └─ Stage 5.  Uncertainty quantification & reporting
                └─ Bootstrap confidence intervals
                └─ JSON + PDF annotated report
```

---

## 3. What Has Been Done

### 3.1 Project Infrastructure

A fully organised project directory has been created with all standard bioinformatics conventions:

```
Facial_Approximation_Using_Forensics/
├── config.yaml              ← Centralised path config for all data sources
├── requirements.txt         ← All Python dependencies with versions
├── setup_env.sh             ← Automated conda environment setup script
├── .gitignore               ← Excludes large genomic files, model weights, etc.
├── data/
│   ├── raw/indigenomes/     ← Raw input VCFs
│   ├── processed/           ← QC'd and transformed data
│   └── reference/           ← Reference panels and SNP lists
├── models/                  ← Trained model artefacts
├── scripts/                 ← Pipeline scripts (one per stage)
├── notebooks/               ← Jupyter notebooks for exploration
├── outputs/                 ← Generated composites and reports
├── docs/                    ← Manuscript and documentation
└── tests/                   ← Unit and integration tests
```

### 3.2 Environment Setup Script (`setup_env.sh`)

An idempotent bash script was written that:

- Creates a conda environment named `dna_facial` with Python 3.11
- Installs bioinformatics CLI tools via conda-forge: `bcftools`, `samtools`, `plink2`, `tabix`
- Installs all Python packages from `requirements.txt` via pip
- Auto-detects conda across 14 common install locations (handles the case where conda is not on PATH in non-interactive shells)
- Verifies every tool and Python library after installation, prints a green success or red failure report

### 3.3 Python Dependencies (`requirements.txt`)

All required packages are pinned with minimum versions:

| Package | Purpose |
|---------|---------|
| `scikit-allel` | VCF parsing, PCA, allele frequency calculations |
| `cyvcf2` | Fast VCF streaming |
| `pysam` | BAM/VCF handling |
| `pandas`, `numpy`, `scipy` | Core scientific computing |
| `scikit-learn` | Random Forest ancestry classifier, PRS models |
| `opencv-python` | Image processing, face warping |
| `dlib` | 68-point facial landmark detection |
| `matplotlib` | Visualisation |
| `requests`, `tqdm` | HTTP downloads with progress bars |
| `pyyaml` | Config file parsing |

### 3.4 Centralised Configuration (`config.yaml`)

A single YAML config file holds all data source paths. This means changing a file path only requires editing one place. Sections cover:

- Reference panel paths (1000 Genomes VCF directory, sample panel)
- IndiGenomes allele frequency table path
- SNP list paths (HIrisPlex-S, facial morphology GWAS)
- GWAS summary statistics directory
- Model artefact paths
- All input/output directory paths

### 3.5 Data Acquisition

#### IndiGenomes Data

The IndiGenomes portal ([clingen.igib.res.in/indigen](http://clingen.igib.res.in/indigen/)) was investigated for data access. The publicly available bulk download is an **Alu mobile element insertion VCF** (`Indigen_Alu_final_geno10_all_22K.vcf`), which has been downloaded and is stored in `data/raw/indigenomes/`. This file contains:

- 22,109 Alu retrotransposon insertion sites
- 1,029 Indian individuals (IND1–IND1029)
- Genome build: GRCh38
- Generated by: PLINK v2.00

This file is **not directly usable** for SNP-based phenotype prediction (it contains structural variants, not SNPs), but it confirms the 1,029-individual IndiGenomes cohort and will be used for structural variant context.

The individual-level SNP VCF for IndiGenomes requires a formal data access agreement with CSIR-IGIB. The Indian-specific allele frequencies are instead being obtained via the `query_indigenomes.py` script (see Section 3.5.3).

#### 1000 Genomes Phase 3 Reference Panel

The **1000 Genomes Phase 3** dataset has been adopted as the primary reference panel. The sample panel file has been successfully downloaded:

- **URL:** [https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel](https://ftp.1000genomes.ebi.ac.uk/vol1/ftp/release/20130502/integrated_call_samples_v3.20130502.ALL.panel)
- **Location:** `data/reference/1000g_panel.txt`
- **Total samples:** 2,504 unrelated individuals
- **South Asian (SAS) samples:** 489 individuals across 5 populations:

| Code | Population | Relevance |
|------|-----------|-----------|
| BEB | Bengali in Bangladesh | Closest to East Indian cohort |
| GIH | Gujarati Indians in Houston, TX | West Indian |
| ITU | Indian Telugu in the UK | South Indian |
| PJL | Punjabi in Lahore, Pakistan | North Indian |
| STU | Sri Lankan Tamil in the UK | South Indian |

Per-chromosome VCF files (genotype data) are queued for download. Priority chromosomes identified:

| Chr | Key loci | Size |
|-----|---------|------|
| 15 | HERC2/OCA2 — eye colour (major HIrisPlex locus) | ~700 MB |
| 16 | MC1R — hair and skin colour | ~600 MB |
| 5 | SLC45A2 — skin pigmentation | ~1.1 GB |
| 2 | EDAR — hair morphology, facial structure | ~1.5 GB |
| 20 | Facial morphology GWAS loci | ~400 MB |
| 22 | Test chromosome (small, good for PCA validation) | ~300 MB |

### 3.6 Pipeline Scripts Written

Three data acquisition scripts are complete and tested:

#### `scripts/download_indigenomes.py`
- Downloads the IndiGenomes VCF from the portal
- Validates the file is a proper gzipped VCF (reads first 5 header lines with cyvcf2)
- Scans the entire file and logs: total variant count, chromosome distribution, AF annotation status
- Saves a summary to `data/processed/indigenomes_summary.json`
- Handles HTTP errors and **resumes interrupted downloads** using HTTP Range headers
- Supports `--local` flag to skip download and validate an existing file
- Supports `--skip-scan` for quick validation only

#### `scripts/download_reference_data.py`
- Downloads 1000 Genomes Phase 3 sample panel and per-chromosome VCFs
- Analyses SAS population breakdown upon panel download
- Resume-capable downloads with tqdm progress bars
- Supports `--chr` flag to download specific chromosomes only
- Supports `--panel-only` to fetch only the sample panel (fast, ~25 KB)
- Supports `--dry-run` to print URLs without downloading
- Saves `data/processed/reference_panel_summary.json`

#### `scripts/query_indigenomes.py`
- Queries the IndiGenomes web portal per-variant endpoint for Indian allele frequencies
- Queries the **Ensembl REST API** in batches for global, SAS (1000 Genomes), and gnomAD South Asian allele frequencies
- Covers **61 target SNPs**: 41 HIrisPlex-S pigmentation SNPs + 20 facial morphology GWAS loci
- Outputs `data/reference/indian_allele_frequencies.tsv` with columns: rsid, chrom, pos_grch38, ref, alt, af_global, af_sas, af_sas_1kg, af_gnomad_sas, af_indigen
- Supports `--snp-list` to query a custom SNP file
- Supports `--source ensembl` to use only the Ensembl API (faster, more reliable)

### 3.7 Living Methods Document (`docs/methods.md`)

A structured methods document has been created in IEEE Access format with seven sections. Each section has a defined scope, full parameter checklist, and TODO markers to be filled in as each pipeline stage is built:

| Section | Title |
|---------|-------|
| III-A | Input Data and Preprocessing |
| III-B | Ancestry Inference — Stage 1: PCA |
| III-C | Ancestry Inference — Stage 2: Random Forest |
| III-D | Pigmentation Prediction — HIrisPlex-S |
| III-E | Morphology Estimation — Ancestry-Weighted PRS |
| III-F | Face Template Warping |
| III-G | Uncertainty Quantification and Output |

---

## 4. Current Data Status

| Dataset | Status | Location |
|---------|--------|---------|
| IndiGenomes Alu VCF (22K structural variants, 1029 individuals) | Downloaded | `data/raw/indigenomes/` |
| 1000 Genomes sample panel (2504 samples, metadata only) | Downloaded | `data/reference/1000g_panel.txt` |
| 1000 Genomes per-chromosome VCFs (genotype data) | Pending download | `data/reference/1000g_vcf/` |
| Indian allele frequencies (Ensembl + IndiGenomes scrape) | Ready to run | `scripts/query_indigenomes.py` |
| HIrisPlex-S SNP list | Pending | `data/reference/hirisplex_s_snp_list.txt` |
| Facial morphology GWAS summary statistics | Pending | `data/reference/` |

---

## 5. Next Steps

The immediate next steps in order of priority:

1. **Download chr 22 VCF** — smallest file (~300 MB), used to validate the full PCA pipeline end-to-end before committing to larger downloads
2. **Run `query_indigenomes.py --source ensembl`** — populate the Indian allele frequency table for all 61 target SNPs
3. **Download HIrisPlex-S SNP list** from the Erasmus MC HIrisPlex-S tool
4. **Build Stage 1 — QC and PCA script** (`scripts/qc_and_ancestry_pca.py`)
5. **Build Stage 2 — HIrisPlex-S prediction script** (`scripts/predict_pigmentation.py`)

---

## 6. Technical Notes

- All scripts are written in **Python 3.11** and follow PEP 8 conventions
- All downloads are **resumable** — interrupted downloads continue from the last byte
- The project is structured so each script can be run independently or chained
- The `config.yaml` is the single source of truth for all file paths
- Large genomic files (VCFs, BAMs, FASTAs) are excluded from git via `.gitignore`
- The conda environment (`dna_facial`) is fully reproducible via `setup_env.sh`
- The methods document is updated incrementally — one section per pipeline stage built

---

*Document auto-generated from project state as of April 2026.*

# DNA-Based Facial Approximation for Indian Forensic Contexts

A bioinformatics pipeline that takes SNP genotype data (VCF format) and produces an annotated 2D facial composite tailored to Indian population genetics. The system leverages genome-wide association studies (GWAS) for facial morphology, pigmentation prediction models (HIrisPlex), and anthropometric references specific to South Asian populations.

## Motivation

Forensic facial approximation from skeletal remains is routinely performed by artists and anthropologists, but DNA-based phenotyping offers an objective, reproducible complement. Most existing tools are calibrated on European cohorts. This project adapts the pipeline for **Indian forensic contexts** by incorporating:

- **IndiGenomes** and the **1000 Genomes South Asian super-population** as reference panels
- GWAS loci discovered or replicated in South Asian samples
- Anthropometric priors reflecting Indian craniofacial variation

## Pipeline Overview

```
VCF (SNP genotypes)
  │
  ├─ 1. Quality control & ancestry inference (COMPLETED)
  │     ├─ PLINK QC & LD Pruning
  │     ├─ PCA against 1000 Genomes (SAS vs EUR)
  │     └─ Super-population classification & ANI/ASI breakdown via Random Forest
  │
  ├─ 2. Pigmentation prediction (IN PROGRESS)
  │     ├─ Extraction of HIrisPlex-S 41-SNPs with strand-flip handling (COMPLETED)
  │     └─ HIrisPlex-S multinomial model → eye, hair, skin colour (TODO)
  │
  ├─ 3. Facial morphology prediction (TODO)
  │     └─ GWAS effect-size scoring → landmark displacements
  │
  ├─ 4. Composite generation (TODO)
  │     └─ Base face + landmark warping + pigmentation overlay
  │
  └─ 5. Annotation & reporting (TODO)
        └─ Confidence intervals, ancestry context, PDF report
```

## Directory Structure

```
.
├── config.yaml              # Paths to all external data sources
├── data/
│   ├── raw/                 # Original unmodified input files (VCFs, etc.)
│   ├── processed/           # Cleaned, filtered, and transformed data (PCA, subject VCFs)
│   └── reference/           # Reference panels (1000G), SNP lists (HIrisPlex), allele frequencies
├── models/                  # Trained models (Random Forest for ancestry)
├── scripts/                 # Pipeline scripts (Data Prep, Clustering, ML, SNP extraction)
├── outputs/                 # Generated plots, experiment reports, JSON inferences, and CSVs
└── README.md
```

## How to Run & Demo the Current Pipeline

The current working pipeline covers Data Preparation, Ancestry Inference, and HIrisPlex SNP Extraction.

### 1. Ancestry Inference & Population Structure

**Prepare the Data (PCA via PLINK):**
```bash
./scripts/pca_prep.sh
```
*(Performs QC, LD Pruning, and calculates top 20 PCs on the 1000 Genomes dataset.)*

**Run Clustering Experiments:**
```bash
python scripts/clustering_experiments.py
python scripts/clustering_experiments_advanced.py
```
*(Generates t-SNE/UMAP plots and silhouette scores comparing K-Means and GMM across populations. Check `outputs/experiments_advanced/` for results.)*

**Run Ancestry Prediction (Random Forest):**
```bash
python scripts/stage2_random_forest.py
```
*(Trains a Random Forest classifier on genomic dosages and predicts ancestry proportions (e.g., ANI/ASI) for a subject. Output is saved to `outputs/ancestry_stage2.json`.)*

### 2. HIrisPlex Phenotype Preparation

**Extract a Subject from the Merged VCF:**
```bash
python scripts/make_subject_vcf.py --sample HG01583
```
*(Scans the massive 1000G merged VCF using Python/gzip and quickly extracts only the 41 HIrisPlex SNPs for the specified subject into `data/processed/subject_hirisplex.vcf`.)*

**Process Subject Dosages & Handle Strand Flips:**
```bash
python scripts/extract_hirisplex_snps.py data/processed/subject_hirisplex.vcf
```
*(Parses the subject VCF, maps alleles against the HIrisPlex template, corrects for Watson-Crick strand flips, checks coverage, and outputs `outputs/hirisplex_dosages.csv`.)*

## Key Data Sources

| Source                                                       | Description                                       |
| ------------------------------------------------------------ | ------------------------------------------------- |
| [IndiGenomes](https://clingen.igib.res.in/indigen/)          | Whole-genome sequences of 1,029 healthy Indians   |
| [1000 Genomes Phase 3](https://www.internationalgenome.org/) | Global reference panel (SAS + EUR super-populations) |
| [HIrisPlex-S](https://hirisplex.erasmusmc.nl/)               | SNP set for eye, hair, and skin colour prediction |
| GWAS Catalog                                                 | Summary statistics for facial morphology loci     |

## Requirements

- Python 3.9+
- `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `umap-learn`
- System libraries: `plink` (1.9)
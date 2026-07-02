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

## How to Run & Demo the Full Pipeline

The full demo now runs from a VCF input all the way to the final composite image.

### 1. Prepare the subject VCF

If you already have a subject-specific VCF, you can use it directly.

If you have a merged VCF and want to extract one sample:
```bash
python scripts/make_subject_vcf.py --sample HG01583 --merged-vcf data/processed/1kg_SAS_EUR_merged.vcf.gz --output data/processed/subject_hirisplex.vcf
```

### 2. Run the full end-to-end pipeline

```bash
python run_pipeline.py --vcf data/processed/subject_hirisplex.vcf
```

If you started from a merged VCF and want the pipeline to extract the sample first:
```bash
python run_pipeline.py --vcf data/processed/1kg_SAS_EUR_merged.vcf.gz --sample HG01583
```

This single command executes:
1. ancestry superpopulation classification
2. Stage 2 ancestry sub-classification
3. sex prediction
4. HIrisPlex SNP extraction
5. HIrisPlex phenotype prediction
6. pigmentation uncertainty scoring
7. morphology PRS scoring
8. morphology trait normalisation
9. template selection
10. landmark detection
11. landmark displacement mapping
12. Delaunay warping
13. pigmentation rendering
14. final forensic composite generation
15. QC report generation

### 3. Final output files

Look in `outputs/` for:
- `final_composite.png`
- `final_composite.pdf`
- `final_colourised_face.png`
- `qc_report.json`
- `qc_report.md`
- `pipeline.log`

## Dashboard UI

Launch the local dashboard to watch the pipeline and inspect every stage:

```bash
python scripts/dashboard_server.py --port 8501
```

Then open `http://127.0.0.1:8501` in your browser.

The dashboard shows:
- a run form for selecting the VCF and sample
- live stage-by-stage pipeline status
- verification checks for each expected artifact
- the pipeline log tail
- the final composite preview

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

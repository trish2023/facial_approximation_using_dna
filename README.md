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
  ├─ 1. Quality control & ancestry inference
  │     └─ PCA against 1000 Genomes + IndiGenomes
  │
  ├─ 2. Pigmentation prediction
  │     └─ HIrisPlex-S model → eye, hair, skin colour
  │
  ├─ 3. Facial morphology prediction
  │     └─ GWAS effect-size scoring → landmark displacements
  │
  ├─ 4. Composite generation
  │     └─ Base face + landmark warping + pigmentation overlay
  │
  └─ 5. Annotation & reporting
        └─ Confidence intervals, ancestry context, PDF report
```

## Directory Structure

```
.
├── config.yaml              # Paths to all external data sources
├── data/
│   ├── raw/                 # Original unmodified input files (VCFs, etc.)
│   ├── processed/           # Cleaned, filtered, and transformed data
│   └── reference/           # Reference panels, SNP lists, allele frequencies
├── models/                  # Trained/serialized prediction models
├── scripts/                 # Pipeline scripts (one per stage)
├── notebooks/               # Exploratory Jupyter notebooks
├── outputs/                 # Generated facial composites and reports
├── docs/                    # Project documentation and references
├── tests/                   # Unit and integration tests
├── requirements.txt         # Python dependencies
└── .gitignore
```

## Quickstart

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd Facial_Approximation_Using_Forensics

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows
pip install -r requirements.txt

# 3. Edit config.yaml with paths to your local data sources

# 4. Run the pipeline on a sample VCF
python scripts/run_pipeline.py --vcf data/raw/sample.vcf.gz --out outputs/
```

## Key Data Sources


| Source                                                       | Description                                       |
| ------------------------------------------------------------ | ------------------------------------------------- |
| [IndiGenomes](https://clingen.igib.res.in/indigen/)          | Whole-genome sequences of 1,029 healthy Indians   |
| [1000 Genomes Phase 3](https://www.internationalgenome.org/) | Global reference panel (SAS super-population)     |
| [HIrisPlex-S](https://hirisplex.erasmusmc.nl/)               | SNP set for eye, hair, and skin colour prediction |
| GWAS Catalog                                                 | Summary statistics for facial morphology loci     |


## Requirements

- Python 3.9+
- System libraries: `libhts` (for pysam/cyvcf2), `dlib` prerequisites (cmake, libboost)
- See `requirements.txt` for the full Python dependency list

## License

This project is intended for academic and forensic research purposes only.
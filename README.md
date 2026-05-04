# DNA-Based Facial Approximation for Indian Forensic Contexts

## Overview

This pipeline predicts externally visible characteristics (EVCs) from SNP genotype data
and generates an annotated 2D facial composite. It is designed for Indian forensic contexts,
leveraging population-specific allele frequencies from the IndiGenomes project and
integrating the HIrisPlex-S pigmentation model with GWAS-derived facial morphology associations.

## Pipeline Summary

```
VCF Input (SNP genotypes)
        |
        v
 1. Preprocessing & QC
    - Filter SNPs, impute missing genotypes
    - Ancestry inference using 1000 Genomes reference panel
        |
        v
 2. Trait Prediction
    - Pigmentation (eye, hair, skin) via HIrisPlex-S model
    - Facial morphology (face shape, nose, lips) via GWAS-trained models
        |
        v
 3. Facial Composite Generation
    - Parametric face synthesis using predicted trait scores
    - Annotation overlay with confidence intervals
        |
        v
 Annotated 2D Facial Composite (PNG/SVG)
```

## Directory Structure

```
dna_facial_approx/
+-- data/
|   +-- raw/            # Unmodified input VCFs and downloaded database files
|   +-- processed/      # QC-filtered, normalised genotype matrices
|   +-- reference/      # HIrisPlex SNP list, 1KG panel, face templates
+-- models/             # Trained classifier and regression model weights
+-- scripts/            # Pipeline scripts (preprocessing, prediction, rendering)
+-- notebooks/          # Exploratory and validation Jupyter notebooks
+-- outputs/            # Final facial composites and per-sample trait reports
+-- docs/               # Methods documentation and data provenance notes
+-- tests/              # Unit and integration tests
```

## Data Sources

| Dataset | Purpose |
|---------|---------|
| IndiGenomes | Indian population allele frequencies and haplotype reference |
| 1000 Genomes Project | Ancestry inference reference panel (SAS + EUR superpopulations) |
| HIrisPlex-S | Eye, hair, and skin colour prediction (41 SNP weights) |
| GWAS summary statistics | Facial morphology trait associations |

Configure all paths in `config.yaml` before running any scripts.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1. Preprocess input VCF
python scripts/preprocess.py --input data/raw/sample.vcf.gz --config config.yaml

# 2. Run trait prediction
python scripts/predict_traits.py --config config.yaml

# 3. Generate facial composite
python scripts/render_composite.py --config config.yaml
```

## Ethical Considerations

Forensic DNA phenotyping outputs are probabilistic estimates intended to assist
investigation, not to serve as definitive identifications. Population-level models may
carry biases. Results should be interpreted by trained forensic practitioners in
compliance with applicable Indian forensic and data-protection legislation.

## License

Research use only. See `docs/` for data provenance and licensing details of
individual datasets.

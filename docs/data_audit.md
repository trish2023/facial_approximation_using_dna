# Data Audit Report

**Generated:** 2026-05-04 14:58  
**Pipeline:** DNA Facial Approximation — Indian forensic context

> *Fast mode (`--skip-vcf-scan`): variant counts and rsID overlap are omitted.*


## 1. Dataset Inventory

File size, variant count, sample count, and inferred genome build for each dataset.

| Dataset | File | Size | Variants / SNPs | Samples | Build | Status |
| ------- | ---- | ---- | --------------- | ------- | ----- | ------ |
| Input VCF (subject) | `sample.vcf.gz` | N/A | N/A | N/A | unknown | *file not found* |
| IndiGenomes | `indigenomes.vcf.gz` | N/A | N/A | N/A | unknown | *file not found* |
| 1000 Genomes SAS+EUR | `1kg_SAS_EUR_merged.vcf.gz` | N/A | N/A | N/A | unknown | *file not found* |
| HIrisPlex-S SNP list | `hirisplex_41snps.csv` | 1.1 KB | 41 SNPs | N/A | N/A | OK |
| GWAS sumstats | `gwas/` | 66.0 MB | 1 files | N/A | N/A | OK |


## 2. HIrisPlex-S SNP Coverage (41 SNPs vs 1000G SAS VCF)

> ⚠️ **1000 Genomes VCF not found — HIrisPlex overlap analysis skipped.**


## 3. GWAS Facial Morphology SNP Coverage

GWS threshold: P < 5e-08.  LD filter: physical clumping ±500 kb (proxy for r² — no LD matrix available at audit time).

> ⚠️ **GWAS files present but no GWS hits extracted (check column names / format).**


## 4. Genome Build Consistency

No VCF files found — cannot assess build consistency.


## 5. Pipeline Readiness Checklist

|  | Requirement | Notes |
| --- | ----------- | ----- |
| ❌ | Input subject VCF | entry point for all stages |
| ❌ | IndiGenomes VCF | population allele-frequency reference |
| ❌ | 1000 Genomes SAS+EUR VCF | ancestry PCA reference panel |
| ✅ | HIrisPlex-S SNP list CSV | 41-SNP pigmentation manifest |
| ✅ | HIrisPlex-S coefficients JSON | model weights — fill TODOs from Walsh 2017 |
| ✅ | GWAS sumstats (≥1 file) | Xiong 2025 / Du 2025 |
| ❌ | GWAS trait index CSV | built by download_gwas_sumstats.py |
| ❌ | Face template directory | base images + landmark annotations |
| ✅ | outputs/ directory | writable output location |
| ✅ | data/processed/ directory | intermediate file store |

**5/10 requirements met (50% ready)**


### Stage-level Summary

|  | Stage | Blocking Issues |
| --- | ----- | --------------- |
| ❌ | Stage 1 · Preprocessing | Missing: sample.vcf.gz, indigenomes.vcf.gz |
| ❌ | Stage 2 · Ancestry Inference (PCA) | Missing: sample.vcf.gz, 1kg_SAS_EUR_merged.vcf.gz |
| ✅ | Stage 3 · Pigmentation (HIrisPlex-S) | — |
| ❌ | Stage 4 · Morphology PRS | Missing: gwas_trait_index.csv |
| ❌ | Stage 5 · Face Rendering | Missing: face_templates |

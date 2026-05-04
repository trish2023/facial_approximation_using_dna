# Methods

*Living document for IEEE Access submission.*
*Each section is a placeholder -- fill in after the corresponding pipeline stage is built and validated.*
*Do not remove TODO markers until the section is finalised for submission.*

---

## 1. Input Data and Preprocessing

### 1.1 Datasets

<!-- TODO: Fill in after preprocessing script is complete.
     Record:
     - IndiGenomes cohort size (N samples, N SNPs before/after QC)
     - 1000 Genomes subset used (superpopulations, N samples)
     - HIrisPlex-S SNP manifest version and source DOI
     - GWAS summary statistics: trait names, source studies, N SNPs per trait
-->

**TODO** -- dataset descriptions, versions, and access dates.

### 1.2 VCF Normalisation and Quality Control

<!-- TODO: Fill in after preprocessing script is complete.
     Record:
     - bcftools norm parameters (left-alignment, multiallelic split)
     - SNP-level filters: call rate threshold, MAF threshold, HWE p-value
     - Sample-level filters: missingness, relatedness (kinship threshold, tool used)
     - Number of SNPs retained after each filter step (attrition table)
-->

**TODO** -- QC filter thresholds, attrition counts at each step.

### 1.3 Genotype Matrix Construction

<!-- TODO: Fill in after preprocessing script is complete.
     Record:
     - Encoding used (0/1/2 dosage vs hard calls)
     - Missing genotype strategy (mean imputation / reference-panel imputation)
     - Final matrix dimensions fed to each downstream module
-->

**TODO** -- encoding scheme, imputation strategy, final matrix dimensions.

---

## 2. Ancestry Inference -- Stage 1: PCA

### 2.1 Reference Panel Preparation

<!-- TODO: Fill in after ancestry PCA script is complete.
     Record:
     - 1KG populations included and sample counts per population
     - SNP overlap between query VCF and reference panel
     - LD pruning parameters (window size, step, r-squared threshold)
     - plink2 flags used verbatim
-->

**TODO** -- reference panel composition, LD pruning parameters.

### 2.2 PCA Projection

<!-- TODO: Fill in after ancestry PCA script is complete.
     Record:
     - Number of PCs computed
     - Whether PCA was fit on reference only and query samples projected in
     - Tool and version (e.g., scikit-allel randomized_pca, plink2 --pca)
     - Variance explained by each PC (table or figure reference)
-->

**TODO** -- PC count, projection method, variance explained.

### 2.3 Visualisation and Sanity Check

<!-- TODO: Fill in after ancestry PCA script is complete.
     Record:
     - PC axes plotted (PC1 vs PC2, PC1 vs PC3)
     - How query samples cluster relative to reference superpopulations
     - Any outlier samples removed at this stage and the criterion used
-->

**TODO** -- PC plot description, clustering observations, outliers removed.

---

## 3. Ancestry Inference -- Stage 2: Random Forest Classifier

### 3.1 Training Set Construction

<!-- TODO: Fill in after ancestry RF script is complete.
     Record:
     - Labels used (superpopulation vs subpopulation level)
     - Class sizes and any resampling applied (SMOTE, class_weight balanced)
     - Train/validation split strategy (k-fold, stratified)
-->

**TODO** -- label granularity, class sizes, split strategy.

### 3.2 Model Architecture and Hyperparameters

<!-- TODO: Fill in after ancestry RF script is complete.
     Record:
     - n_estimators, max_depth, min_samples_leaf, max_features
     - Hyperparameter search method (GridSearchCV / RandomizedSearchCV) and CV folds
     - Final chosen hyperparameters
-->

**TODO** -- RF hyperparameters and tuning procedure.

### 3.3 Performance Evaluation

<!-- TODO: Fill in after ancestry RF script is complete.
     Record:
     - Accuracy, macro F1, confusion matrix on held-out set
     - Per-class precision and recall (table)
     - Predicted ancestry probability vector format passed to downstream modules
-->

**TODO** -- classification metrics, confusion matrix, output format.

---

## 4. Pigmentation Prediction -- HIrisPlex-S

### 4.1 SNP Extraction and Allele Harmonisation

<!-- TODO: Fill in after HIrisPlex prediction script is complete.
     Record:
     - Number of HIrisPlex-S SNPs present in query VCF (out of 41)
     - Strategy for missing HIrisPlex SNPs (mean dosage, reference allele imputation)
     - Strand alignment and allele-flipping procedure
-->

**TODO** -- SNP coverage, missing SNP handling, strand harmonisation.

### 4.2 Multinomial Logistic Regression Scoring

<!-- TODO: Fill in after HIrisPlex prediction script is complete.
     Record:
     - Traits predicted: eye colour (categories), hair colour (categories), skin colour (ITA/FST scale)
     - Coefficient source (HIrisPlex-S paper DOI, supplementary table number)
     - Softmax output format: per-category probabilities
     - Validation: AUC on any available Indian samples with known phenotypes
-->

**TODO** -- model coefficient source, output probability format, validation results.

### 4.3 Population-Frequency Adjustment

<!-- TODO: Fill in after HIrisPlex prediction script is complete.
     Record:
     - Whether IndiGenomes allele frequencies were used to adjust priors
     - Method (Bayesian update vs frequency-weighted softmax)
     - Effect size of adjustment on predicted probabilities (delta table)
-->

**TODO** -- prior adjustment method and magnitude.

---

## 5. Morphology Estimation -- Ancestry-Weighted PRS

### 5.1 GWAS Summary Statistics Curation

<!-- TODO: Fill in after morphology PRS script is complete.
     Record:
     - Traits modelled (list each: nose width, face width, lip thickness, etc.)
     - Source GWAS per trait: first author, year, PMID, N_discovery
     - SNP count per trait after clumping and p-value threshold
     - Clumping parameters (r-squared, physical window, p-value threshold)
-->

**TODO** -- trait list, GWAS sources, clumping parameters per trait.

### 5.2 Polygenic Score Computation

<!-- TODO: Fill in after morphology PRS script is complete.
     Record:
     - PRS formula (weighted sum of effect-allele dosages x beta)
     - Tool used (plink2 --score flags, or custom numpy implementation)
     - Score normalisation (z-score relative to IndiGenomes reference distribution)
-->

**TODO** -- PRS formula, tool/flags, normalisation scheme.

### 5.3 Ancestry Weighting

<!-- TODO: Fill in after morphology PRS script is complete.
     Record:
     - How the ancestry probability vector from Stage 2 modulates PRS
     - Whether separate GWAS betas per superpopulation were used (SAS vs EUR)
     - Weighted combination formula
-->

**TODO** -- ancestry-weighting formula and population-specific beta sources.

### 5.4 Trait Score Validation

<!-- TODO: Fill in after morphology PRS script is complete.
     Record:
     - Correlation of PRS with measured traits in any available validation cohort
     - Comparison to European-trained PRS baseline (SAS transferability delta)
-->

**TODO** -- cross-population PRS transferability metrics.

---

## 6. Face Template Warping

### 6.1 Template Library

<!-- TODO: Fill in after rendering script is complete.
     Record:
     - Source of base face templates (synthetic, photofit, or parametric model)
     - Number of templates and demographic stratification (sex, age bracket)
     - Landmark annotation scheme (N landmarks, annotation tool used)
     - Licensing / ethical approval for template use
-->

**TODO** -- template source, count, landmark scheme, licensing.

### 6.2 Trait-to-Landmark Mapping

<!-- TODO: Fill in after rendering script is complete.
     Record:
     - Mapping from each predicted trait score to landmark displacement vector
     - Whether mapping is linear (regression coefficients) or non-linear (thin-plate spline)
     - Source of mapping coefficients (empirical or published 3D morphometrics study)
-->

**TODO** -- trait-to-landmark mapping method and coefficient source.

### 6.3 Warp Procedure

<!-- TODO: Fill in after rendering script is complete.
     Record:
     - Warping algorithm (thin-plate spline, affine, dlib shape predictor)
     - OpenCV / dlib functions called with parameters
     - Colour and texture application method for pigmentation traits
-->

**TODO** -- warping algorithm, library calls, pigmentation rendering.

---

## 7. Uncertainty Quantification and Output

### 7.1 Per-Trait Confidence Intervals

<!-- TODO: Fill in after uncertainty module is complete.
     Record:
     - CI method for pigmentation (Dirichlet posterior vs bootstrap)
     - CI method for morphology PRS (parametric SE vs bootstrap percentile)
     - Confidence level (default 95%; record if changed)
-->

**TODO** -- CI method per trait, confidence level.

### 7.2 Composite Uncertainty Propagation

<!-- TODO: Fill in after uncertainty module is complete.
     Record:
     - How per-trait CIs propagate to landmark displacement uncertainty
     - Whether composite shows a single MAP estimate or a warp range overlay
-->

**TODO** -- uncertainty propagation to composite image.

### 7.3 Output Artefacts

<!-- TODO: Fill in after rendering and reporting scripts are complete.
     Record:
     - Output image format and resolution (px)
     - Per-sample JSON/CSV report fields (trait probabilities, CIs, ancestry vector)
     - Annotation schema overlaid on composite image
-->

**TODO** -- output file formats, report schema, annotation layout.

---

## Appendix: Parameter Change Log

*Record every parameter decision here as the pipeline is built. This log becomes the audit trail for the Methods section and reviewer responses.*

| Date | Section | Parameter | Value | Reason |
|------|---------|-----------|-------|--------|
| --   | --      | --        | --    | --     |

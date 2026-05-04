# Methods

> **Living document** — each section is filled in after the corresponding pipeline
> stage is implemented and validated. Sections marked **TODO** are placeholders.
> Written in IEEE Access style; intended to become §&nbsp;III of the manuscript.

---

## III-A. Input Data and Preprocessing

<!-- TODO: Fill after pipeline stage 1 is built -->

**Scope.** Describe the input VCF specification, quality-control filters applied
to raw genotypes, and any site- or sample-level exclusions.

- **Input format:** TODO — VCF version, single- vs. multi-sample, expected caller
- **QC filters:**
  - Genotype quality (GQ) threshold: `TODO`
  - Read depth (DP) threshold: `TODO`
  - Minor allele frequency (MAF) cutoff: `TODO`
  - SNP missingness rate cutoff: `TODO`
  - Hardy–Weinberg equilibrium p-value threshold: `TODO`
- **Reference alignment:** TODO — genome build (GRCh37 / GRCh38), liftover strategy if needed
- **Variant intersection:** TODO — how input SNPs are intersected with the analysis panel (IndiGenomes + 1000 Genomes)
- **Dataset sizes:**
  - IndiGenomes samples: `TODO`
  - 1000 Genomes Phase 3 SAS samples: `TODO`
  - Total SNPs retained after QC: `TODO`
- **Software and versions:** TODO

---

## III-B. Ancestry Inference — Stage 1: Principal Component Analysis

<!-- TODO: Fill after PCA module is built -->

**Scope.** Project the query sample into a reference PCA space to obtain a
continuous ancestry representation used downstream for population-aware scoring.

- **Reference panel composition:** TODO — populations included, sample counts per population
- **LD pruning parameters:**
  - Window size: `TODO`
  - Step size: `TODO`
  - r² threshold: `TODO`
  - SNPs retained after pruning: `TODO`
- **PCA method:** TODO — scikit-allel `randomized_pca` / `sklearn.decomposition.PCA`
- **Number of principal components retained:** `TODO`
- **Variance explained by retained PCs:** `TODO`
- **Projection method for query sample:** TODO — describe how a single new sample is projected into the reference PC space
- **Visualisation:** TODO — reference to supplementary PCA scatter plot
- **Software and versions:** TODO

---

## III-C. Ancestry Inference — Stage 2: Random Forest Classification

<!-- TODO: Fill after RF classifier is built -->

**Scope.** Classify the query sample into a population group (or admixture
proportions) using the PCA coordinates from Stage 1.

- **Training labels:** TODO — population labels used (e.g., 1000 Genomes superpopulation codes, IndiGenomes state-level or linguistic groups)
- **Feature vector:** TODO — which PCs are used as features
- **Model:**
  - Algorithm: Random Forest (`sklearn.ensemble.RandomForestClassifier`)
  - Number of estimators: `TODO`
  - Max depth: `TODO`
  - Class weighting strategy: `TODO`
  - Other hyperparameters: `TODO`
- **Training / validation split:** TODO — ratio, stratification
- **Cross-validation:** TODO — k-fold, stratified
- **Performance metrics:**
  - Accuracy: `TODO`
  - Macro F1: `TODO`
  - Confusion matrix: TODO — reference to supplementary table
- **Soft classification output:** TODO — describe how `predict_proba` is used to derive admixture-like proportions for downstream weighting
- **Software and versions:** TODO

---

## III-D. Pigmentation Prediction — HIrisPlex-S

<!-- TODO: Fill after pigmentation module is built -->

**Scope.** Predict eye colour, hair colour, and skin colour from a defined set of
SNPs using the HIrisPlex-S multinomial logistic regression model.

- **SNP panel:**
  - Total HIrisPlex-S SNPs: `TODO`
  - SNPs present in our analysis set: `TODO`
  - Handling of missing SNPs: `TODO`
- **Model implementation:**
  - Source of model weights: TODO — original Erasmus MC coefficients vs. retrained
  - Eye colour categories: `TODO`
  - Hair colour categories: `TODO`
  - Skin colour categories: `TODO`
- **Genotype encoding:** TODO — additive (0/1/2), handling of strand orientation
- **Prediction outputs:**
  - Per-category probabilities for eye, hair, skin: `TODO`
  - Thresholding / argmax strategy: `TODO`
- **Validation on Indian reference samples:** TODO — concordance with self-reported phenotype where available
- **Known limitations for South Asian populations:** TODO
- **Software and versions:** TODO

---

## III-E. Morphology Estimation — Ancestry-Weighted Polygenic Risk Scores

<!-- TODO: Fill after PRS module is built -->

**Scope.** Estimate facial landmark displacements from GWAS effect sizes, weighted
by the ancestry proportions inferred in Stage 2.

- **GWAS summary statistics:**
  - Source: `TODO`
  - Traits: `TODO` (e.g., nose width, face width, jaw angle, brow ridge prominence, lip thickness)
  - Number of loci per trait: `TODO`
  - Genome-wide significance threshold: `TODO`
- **PRS calculation:**
  - Method: TODO — simple weighted sum / clumping + thresholding / LDpred2 / PRScs
  - LD reference panel used: `TODO`
  - Clumping window: `TODO`
  - Clumping r² threshold: `TODO`
  - SNPs included per trait after clumping: `TODO`
- **Ancestry weighting:**
  - Describe how population-specific allele frequencies or effect sizes are combined using the admixture proportions from §&nbsp;III-C: `TODO`
  - Reference populations used for frequency lookup: `TODO`
- **Landmark mapping:** TODO — how PRS values are converted to physical displacements (mm or pixels) on the 68-point facial landmark model
- **Normalisation:** TODO — z-score against reference distribution, scaling factor
- **Software and versions:** TODO

---

## III-F. Face Template Warping

<!-- TODO: Fill after composite generation module is built -->

**Scope.** Generate a 2-D facial composite by warping a neutral base template
according to the predicted landmark displacements and overlaying pigmentation.

- **Base template:**
  - Source: `TODO`
  - Resolution: `TODO`
  - Landmark annotation method: `TODO`
  - Population / sex of base template(s): `TODO`
- **Landmark model:**
  - Detector: TODO — dlib 68-point, MediaPipe, or custom
  - Predictor weights file: `TODO`
- **Warping algorithm:**
  - Method: TODO — Delaunay triangulation + affine warp / thin-plate spline / piecewise affine
  - Implementation: `TODO`
  - Boundary handling: `TODO`
- **Pigmentation overlay:**
  - Skin colour mapping: TODO — RGB / HSV adjustment of base template
  - Eye colour compositing: `TODO`
  - Hair colour rendering strategy: `TODO`
- **Post-processing:** TODO — blending, smoothing, artefact correction
- **Output format:** TODO — resolution, file type, metadata embedded
- **Software and versions:** TODO

---

## III-G. Uncertainty Quantification and Output

<!-- TODO: Fill after reporting module is built -->

**Scope.** Quantify prediction confidence at each stage and present results in an
annotated report suitable for forensic casework.

- **Per-stage confidence metrics:**
  - Ancestry classification: TODO — posterior probability, entropy
  - Pigmentation: TODO — category probability margin
  - Morphology PRS: TODO — percentile rank, standard error of the score
- **Composite-level uncertainty:**
  - Method: TODO — Monte Carlo dropout, bootstrap resampling of effect sizes, or analytic propagation
  - Number of replicates: `TODO`
  - Visualisation of uncertainty: TODO — confidence ellipses on landmarks, heatmap overlay
- **Output artefacts:**
  - Annotated facial composite (PNG): `TODO`
  - Structured prediction report (JSON): `TODO`
  - Human-readable PDF report: `TODO`
  - Fields included: `TODO`
- **Forensic disclaimers:** TODO — standard caveats on DNA phenotyping accuracy, population-specific limitations, not for identification
- **Software and versions:** TODO

---

> **Revision log**
>
> | Date | Section | Change |
> |------|---------|--------|
> | <!-- date --> | — | Initial scaffold created |

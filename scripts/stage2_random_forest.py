"""
stage2_random_forest.py

Train a Random Forest classifier for stage-2 fine-grained South Asian
sub-ancestry classification and apply it to a forensic subject.

Pipeline
--------
1. Load stage-2 training data produced by stage2_training_data.py:
     data/processed/stage2_training/{features.npy, labels.npy,
       label_encoder.pkl, snp_list.csv, metadata.json}
2. GridSearchCV over n_estimators × max_depth × min_samples_leaf
   (18 combos × 5-fold stratified CV).  Best params selected by accuracy;
   macro-F1 is used as a tiebreaker.
3. Refit final RF on full training set; save model bundle to
   models/stage2_rf.pkl.
4. Parse subject VCF (data/raw/sample.vcf.gz) to extract alt-allele
   dosage at the trained SNP loci; missing positions are mean-imputed.
5. Predict sub-population label and per-class probability vector.
6. Derive ANI proportion (alpha) and ASI proportion from the probability
   vector using a population-to-clade taxonomy.
7. Assert that alpha + ASI ≈ 1.0 (soft warning if |sum - 1| > 0.10).
8. Write outputs/ancestry_stage2.json.

Usage
-----
    python scripts/stage2_random_forest.py
    python scripts/stage2_random_forest.py --retrain
    python scripts/stage2_random_forest.py --subject-id CASE001
    python scripts/stage2_random_forest.py --n-jobs 4
    python scripts/stage2_random_forest.py --training-dir data/processed/stage2_training
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# GridSearch parameter grid (exactly as specified)
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "n_estimators":    [100, 500],
    "max_depth":       [None, 10, 20],
    "min_samples_leaf": [1, 3, 5],
}
CV_FOLDS = 5

# ---------------------------------------------------------------------------
# ANI / ASI population taxonomy
#
# Classes whose names match any pattern in ANI_PATTERNS count toward the
# Indo-Aryan (ANI) proportion.  ASI_PATTERNS cover Dravidian-speaking and
# Austro-Asiatic tribal groups (both show high ASI ancestry in STRUCTURE
# analyses).  Unmatched classes are flagged as "other" and excluded from
# the alpha + ASI sum.
# ---------------------------------------------------------------------------
ANI_PATTERNS: tuple[str, ...] = (
    # 1KG proxy label
    "indoaryan",
    # Linguistic / regional
    "gujarati", "punjabi", "bengali", "marathi", "rajasthani",
    "kashmiri", "sindhi", "bihari", "bhojpuri", "odia", "oriya",
    "assamese", "nepali", "sinhala", "konkani", "dogri", "bodo",
    "maithili", "awadhi", "khandeshi",
    # 1KG population codes (raw)
    "gih", "pjl",
)
ASI_PATTERNS: tuple[str, ...] = (
    # 1KG proxy label
    "dravidian",
    # Dravidian-speaking
    "tamil", "telugu", "kannada", "malayalam", "tulu",
    "toda", "kota", "irula", "badaga", "kurumba", "koya",
    "kondh", "gondi", "gond", "kolam", "yerukala", "chenchu",
    # Austro-Asiatic tribal (high ASI ancestry)
    "santali", "mundari", "munda", "ho", "oraon",
    "khasi", "garo", "bhil", "bhili",
    # 1KG population codes (raw)
    "beb", "itu", "stu",
)

ANI_ASI_TOLERANCE = 0.10   # warn if |alpha + ASI - 1.0| exceeds this


def _classify_class(name: str) -> str:
    """Return 'ANI', 'ASI', or 'other' for a population class name."""
    lower = name.lower().replace("-", "").replace("_", "").replace(" ", "")
    for pat in ANI_PATTERNS:
        if pat in lower:
            return "ANI"
    for pat in ASI_PATTERNS:
        if pat in lower:
            return "ASI"
    return "other"


# ===========================================================================
# Data loading
# ===========================================================================

def load_training_data(
    training_dir: Path,
) -> tuple[np.ndarray, np.ndarray, LabelEncoder, pd.DataFrame, dict[str, Any]]:
    """Load all stage-2 training artefacts.

    Returns (X, y, label_encoder, snp_df, training_metadata).
    """
    required = [
        training_dir / "features.npy",
        training_dir / "labels.npy",
        training_dir / "label_encoder.pkl",
        training_dir / "snp_list.csv",
        training_dir / "metadata.json",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        log.error(
            "Missing stage-2 training files:\n%s\n"
            "Run scripts/stage2_training_data.py first.",
            "\n".join(f"  {p}" for p in missing),
        )
        sys.exit(1)

    X   = np.load(training_dir / "features.npy")
    y   = np.load(training_dir / "labels.npy")
    le  = joblib.load(training_dir / "label_encoder.pkl")
    snp = pd.read_csv(training_dir / "snp_list.csv")
    with open(training_dir / "metadata.json") as fh:
        meta = json.load(fh)

    log.info(
        "Training data: %d samples × %d SNPs, %d classes: %s",
        X.shape[0], X.shape[1], len(le.classes_), list(le.classes_),
    )
    return X, y, le, snp, meta


# ===========================================================================
# GridSearchCV
# ===========================================================================

def run_gridsearch(
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int,
    seed: int,
) -> tuple[RandomForestClassifier, pd.DataFrame]:
    """Run GridSearchCV and return (best_estimator, results_dataframe)."""
    n_combos = (
        len(PARAM_GRID["n_estimators"])
        * len(PARAM_GRID["max_depth"])
        * len(PARAM_GRID["min_samples_leaf"])
    )
    log.info(
        "GridSearchCV: %d parameter combinations × %d-fold CV (%d fits).",
        n_combos, CV_FOLDS, n_combos * CV_FOLDS,
    )
    log.info("Parameter grid: %s", PARAM_GRID)
    log.info("This may take several minutes with n_estimators=500 …")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    rf_base = RandomForestClassifier(
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
    )
    gs = GridSearchCV(
        estimator=rf_base,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="accuracy",
        refit=True,
        n_jobs=1,       # parallelise within each RF fit (via n_jobs on estimator)
        verbose=1,
        return_train_score=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gs.fit(X, y)

    results = pd.DataFrame(gs.cv_results_)

    # Also compute macro-F1 for the best param set (for reporting)
    best_idx  = gs.best_index_
    best_acc  = gs.best_score_
    best_params = gs.best_params_

    log.info("Best CV accuracy: %.4f  params=%s", best_acc, best_params)

    # Print top-5 results
    cols = ["param_n_estimators", "param_max_depth", "param_min_samples_leaf",
            "mean_test_score", "std_test_score", "rank_test_score"]
    top5 = results.sort_values("rank_test_score").head(5)[cols]
    print("\n--- GridSearchCV Top 5 ---")
    print(top5.to_string(index=False))
    print()

    return gs.best_estimator_, results


# ===========================================================================
# Final model evaluation and saving
# ===========================================================================

def evaluate_and_save(
    model: RandomForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    le: LabelEncoder,
    snp_df: pd.DataFrame,
    training_meta: dict[str, Any],
    model_path: Path,
    seed: int,
) -> dict[str, Any]:
    """
    Run held-out CV evaluation (same folds), refit on full data, save bundle.
    Returns a summary dict for the output JSON.
    """
    # Cross-validate the final hyperparameters for honest accuracy estimate
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    fold_acc, fold_f1 = [], []
    for fold, (tr, va) in enumerate(cv.split(X, y)):
        m = RandomForestClassifier(
            **model.get_params(),
        )
        m.fit(X[tr], y[tr])
        preds = m.predict(X[va])
        fold_acc.append(accuracy_score(y[va], preds))
        fold_f1.append(f1_score(y[va], preds, average="macro", zero_division=0))

    cv_acc_mean = float(np.mean(fold_acc))
    cv_acc_std  = float(np.std(fold_acc))
    cv_f1_mean  = float(np.mean(fold_f1))
    log.info(
        "Final model CV: accuracy=%.4f±%.4f  macro-F1=%.4f",
        cv_acc_mean, cv_acc_std, cv_f1_mean,
    )

    # Refit on full training set
    model.fit(X, y)
    full_preds = model.predict(X)
    full_acc   = accuracy_score(y, full_preds)
    log.info("Train accuracy (full set, not OOB): %.4f", full_acc)
    log.info("OOB score available: %s", model.oob_score_ if hasattr(model, "oob_score_") else "N/A")

    print("\n--- Classification report (full training set) ---")
    print(classification_report(y, full_preds, target_names=le.classes_, zero_division=0))

    # Feature means for subject imputation
    feature_means = X.mean(axis=0).astype(np.float32)

    # Taxonomy
    class_clade = {cls: _classify_class(cls) for cls in le.classes_}
    log.info("Class taxonomy: %s", class_clade)

    bundle = {
        "model":          model,
        "label_encoder":  le,
        "snp_list":       snp_df,        # DataFrame: CHROM, POS, ID, REF, ALT
        "feature_means":  feature_means,  # (n_snps,) for mean imputation
        "class_clade":    class_clade,    # class_name → 'ANI'|'ASI'|'other'
        "training_meta":  training_meta,
        "cv_accuracy":    {"mean": cv_acc_mean, "std": cv_acc_std},
        "cv_macro_f1":    cv_f1_mean,
        "hyperparameters": model.get_params(),
        "timestamp":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path, compress=3)
    log.info("Model bundle saved → %s", model_path)

    return {
        "cv_accuracy_mean": cv_acc_mean,
        "cv_accuracy_std":  cv_acc_std,
        "cv_macro_f1":      cv_f1_mean,
        "hyperparameters":  {k: (v if v is not None else "None")
                             for k, v in model.get_params().items()
                             if k in PARAM_GRID},
    }


# ===========================================================================
# Subject VCF parsing
# ===========================================================================

def _open_vcf(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")


def _complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)


def parse_subject_dosages(
    vcf_path: Path,
    snp_df: pd.DataFrame,
    feature_means: np.ndarray,
    subject_id: str | None,
) -> tuple[np.ndarray, str, int]:
    """
    Extract alt-allele dosage for each SNP in snp_df from the subject VCF.

    Returns (dosage_vector, resolved_sample_id, n_missing).
    Missing SNPs are imputed with the training-set column mean.

    Matching strategy:
      1. Primary: CHROM (chr-prefix normalised) + POS
      2. If REF/ALT are strand-swapped (complement match), dosage is flipped.
    """
    n_snps = len(snp_df)
    dosage = np.full(n_snps, np.nan, dtype=np.float32)

    # Build lookup: (norm_chrom, pos) → row index in snp_df
    def _norm_chrom(c: str) -> str:
        return c.lower().lstrip("chr")

    snp_df = snp_df.reset_index(drop=True)
    lookup: dict[tuple[str, int], int] = {
        (_norm_chrom(str(row.CHROM)), int(row.POS)): idx
        for idx, row in snp_df.iterrows()
    }

    resolved_sample = subject_id or "SAMPLE"
    sample_col: int | None = None

    with _open_vcf(vcf_path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue

            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                samples = cols[9:]
                if not samples:
                    log.warning("VCF has no sample columns.")
                    break
                if subject_id and subject_id in samples:
                    sample_col = samples.index(subject_id)
                    resolved_sample = subject_id
                elif subject_id:
                    log.warning(
                        "Sample '%s' not in VCF; using first sample '%s'.",
                        subject_id, samples[0],
                    )
                    sample_col = 0
                    resolved_sample = samples[0]
                else:
                    sample_col = 0
                    resolved_sample = samples[0]
                continue

            if sample_col is None:
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue

            chrom_raw = parts[0]
            try:
                pos = int(parts[1])
            except ValueError:
                continue

            key = (_norm_chrom(chrom_raw), pos)
            if key not in lookup:
                continue

            idx = lookup[key]
            ref_train = snp_df.at[idx, "REF"]
            alt_train = snp_df.at[idx, "ALT"]
            ref_vcf   = parts[3]
            alt_vcf   = parts[4]   # only biallelic

            fmt_parts = parts[8].split(":")
            if "GT" not in fmt_parts:
                continue
            gt_idx    = fmt_parts.index("GT")
            samp_data = parts[9 + sample_col].split(":")
            if gt_idx >= len(samp_data):
                continue

            gt_str = samp_data[gt_idx]
            if "." in gt_str:
                continue   # missing → stays NaN → will be mean-imputed

            sep     = "|" if "|" in gt_str else "/"
            alleles = gt_str.split(sep)
            allele_seq = {"0": ref_vcf, "1": alt_vcf}

            # Count copies of alt_train allele
            alt_count = 0
            valid = 0
            for a in alleles:
                seq = allele_seq.get(a)
                if seq is None:
                    valid = 0
                    break
                if seq == alt_train or seq == _complement(alt_train):
                    alt_count += 1
                valid += 1

            if valid == 0:
                continue

            # Strand-flip check: if VCF REF matches training ALT (complement OK),
            # the dosage needs inverting (2 → 0, 1 → 1, 0 → 2).
            flipped = (
                (ref_vcf == alt_train or ref_vcf == _complement(alt_train))
                and (alt_vcf == ref_train or alt_vcf == _complement(ref_train))
            )
            if flipped:
                alt_count = len(alleles) - alt_count

            dosage[idx] = float(alt_count)

    # Mean-impute missing
    n_missing = int(np.isnan(dosage).sum())
    if n_missing > 0:
        dosage = np.where(np.isnan(dosage), feature_means, dosage)

    log.info(
        "Subject SNP extraction: %d / %d positions found  (%d mean-imputed)",
        n_snps - n_missing, n_snps, n_missing,
    )
    return dosage, resolved_sample, n_missing


# ===========================================================================
# ANI / ASI proportion derivation
# ===========================================================================

def derive_ani_asi(
    proba: np.ndarray,
    classes: list[str],
    class_clade: dict[str, str],
) -> dict[str, Any]:
    """
    Sum probabilities by clade membership.

    Returns dict with keys:
        alpha          – ANI proportion (sum of IndoAryan-class probabilities)
        asi            – ASI proportion (sum of Dravidian/tribal-class probs)
        other          – remaining probability (unclassified classes)
        ani_classes    – list of class names counted toward alpha
        asi_classes    – list of class names counted toward asi
        other_classes  – list of class names not assigned to either clade
        alpha_plus_asi – alpha + asi (should be ≈ 1.0)
    """
    ani_prob   = 0.0
    asi_prob   = 0.0
    other_prob = 0.0
    ani_cls: list[str] = []
    asi_cls: list[str] = []
    oth_cls: list[str] = []

    for cls, p in zip(classes, proba):
        clade = class_clade.get(cls, "other")
        if clade == "ANI":
            ani_prob += p
            ani_cls.append(cls)
        elif clade == "ASI":
            asi_prob += p
            asi_cls.append(cls)
        else:
            other_prob += p
            oth_cls.append(cls)

    total = ani_prob + asi_prob + other_prob
    log.info(
        "ANI (alpha)=%.4f  ASI=%.4f  other=%.4f  total=%.4f",
        ani_prob, asi_prob, other_prob, total,
    )

    alpha_plus_asi = ani_prob + asi_prob
    if abs(alpha_plus_asi - 1.0) > ANI_ASI_TOLERANCE:
        log.warning(
            "alpha + ASI = %.4f (expected ≈ 1.0, tolerance ±%.2f). "
            "Unclassified classes: %s. "
            "Check ANI_PATTERNS / ASI_PATTERNS in the script.",
            alpha_plus_asi, ANI_ASI_TOLERANCE, oth_cls,
        )
    else:
        log.info("alpha + ASI = %.4f ✓ (within ±%.2f tolerance)", alpha_plus_asi, ANI_ASI_TOLERANCE)

    return {
        "alpha":          round(float(ani_prob), 6),
        "asi":            round(float(asi_prob), 6),
        "other":          round(float(other_prob), 6),
        "alpha_plus_asi": round(float(alpha_plus_asi), 6),
        "ani_classes":    ani_cls,
        "asi_classes":    asi_cls,
        "other_classes":  oth_cls,
    }


# ===========================================================================
# Prediction
# ===========================================================================

def predict_subject(
    bundle: dict[str, Any],
    dosage: np.ndarray,
    sample_id: str,
    n_missing: int,
) -> dict[str, Any]:
    """Apply the loaded model bundle to a subject dosage vector."""
    model: RandomForestClassifier = bundle["model"]
    le:    LabelEncoder           = bundle["label_encoder"]
    class_clade: dict[str, str]   = bundle["class_clade"]

    X_subj = dosage.reshape(1, -1)
    pred_enc = int(model.predict(X_subj)[0])
    pred_label = le.inverse_transform([pred_enc])[0]
    proba = model.predict_proba(X_subj)[0]

    log.info("Subject prediction: %s  (confidence=%.4f)", pred_label, float(proba.max()))

    prob_vector = {cls: round(float(p), 6) for cls, p in zip(le.classes_, proba)}

    ani_asi = derive_ani_asi(list(proba), list(le.classes_), class_clade)

    return {
        "sample_id":           sample_id,
        "predicted_subpop":    pred_label,
        "predicted_clade":     class_clade.get(pred_label, "other"),
        "confidence":          round(float(proba.max()), 6),
        "probability_vector":  prob_vector,
        "ani_asi_proportions": ani_asi,
        "snps_missing":        n_missing,
        "snps_total":          len(le.classes_),   # note: this is n_classes, not n_snps
    }


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage-2 Random Forest sub-ancestry classifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",       default="config.yaml",
                   help="Pipeline config YAML (default: config.yaml)")
    p.add_argument("--training-dir", default=None,
                   help="Stage-2 training directory (default: data/processed/stage2_training)")
    p.add_argument("--vcf",          default=None,
                   help="Subject VCF (.vcf / .vcf.gz).  Overrides config.")
    p.add_argument("--subject-id",   default=None, dest="subject_id",
                   help="Sample ID in multi-sample VCF (default: first sample).")
    p.add_argument("--model-out",    default=None, dest="model_out",
                   help="Model output path (default: models/stage2_rf.pkl)")
    p.add_argument("--out",          default=None,
                   help="JSON output path (default: outputs/ancestry_stage2.json)")
    p.add_argument("--retrain",      action="store_true",
                   help="Force retraining even if models/stage2_rf.pkl exists.")
    p.add_argument("--skip-subject", action="store_true",
                   help="Train model only; skip subject prediction.")
    p.add_argument("--n-jobs",       type=int, default=-1,
                   help="Parallel jobs for RF fitting (default: -1 = all cores).")
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--debug",        action="store_true")
    return p.parse_args()


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    args = _parse_args()
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg      = _load_config(REPO_ROOT / args.config)
    paths_cfg = cfg.get("paths", {})

    training_dir = (
        Path(args.training_dir) if args.training_dir
        else REPO_ROOT / paths_cfg.get("processed_dir", "data/processed") / "stage2_training"
    )
    vcf_path = (
        Path(args.vcf) if args.vcf
        else REPO_ROOT / paths_cfg.get("input_vcf", "data/raw/sample.vcf.gz")
    )
    model_path = (
        Path(args.model_out) if args.model_out
        else REPO_ROOT / "models" / "stage2_rf.pkl"
    )
    out_path = (
        Path(args.out) if args.out
        else REPO_ROOT / paths_cfg.get("output_dir", "outputs") / "ancestry_stage2.json"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load training data
    # ------------------------------------------------------------------
    X, y, le, snp_df, training_meta = load_training_data(training_dir)

    # ------------------------------------------------------------------
    # Step 2 & 3 — GridSearch + final model
    # ------------------------------------------------------------------
    model_summary: dict[str, Any] = {}

    if model_path.exists() and not args.retrain:
        log.info("Loading existing model bundle: %s  (use --retrain to force retrain)", model_path)
        bundle = joblib.load(model_path)
        model_summary = {
            "source":          "cached",
            "cv_accuracy_mean": bundle.get("cv_accuracy", {}).get("mean"),
            "cv_macro_f1":     bundle.get("cv_macro_f1"),
            "hyperparameters": {k: (v if v is not None else "None")
                                for k, v in bundle.get("hyperparameters", {}).items()
                                if k in PARAM_GRID},
        }
        log.info("Cached model params: %s", bundle.get("hyperparameters", {}))
    else:
        log.info("=== GridSearchCV ===")
        best_model, gs_results = run_gridsearch(X, y, args.n_jobs, args.seed)
        gs_results.to_csv(
            training_dir / "gridsearch_results.csv", index=False,
        )
        log.info("Full GridSearch results → %s", training_dir / "gridsearch_results.csv")

        log.info("=== Evaluating & saving final model ===")
        model_summary = evaluate_and_save(
            best_model, X, y, le, snp_df, training_meta, model_path, args.seed,
        )
        model_summary["source"] = "trained"
        bundle = joblib.load(model_path)

    if args.skip_subject:
        log.info("--skip-subject set; exiting after model training.")
        return

    # ------------------------------------------------------------------
    # Step 4 — Extract subject features
    # ------------------------------------------------------------------
    if not vcf_path.exists():
        log.error(
            "Subject VCF not found: %s\n"
            "  Provide --vcf <path> or --skip-subject to train only.",
            vcf_path,
        )
        sys.exit(1)

    log.info("=== Parsing subject VCF: %s ===", vcf_path)
    feature_means: np.ndarray = bundle["feature_means"]
    snp_df_bundle: pd.DataFrame = bundle["snp_list"]

    dosage, resolved_sample, n_missing = parse_subject_dosages(
        vcf_path, snp_df_bundle, feature_means, args.subject_id,
    )

    # ------------------------------------------------------------------
    # Step 5-6 — Predict + ANI/ASI
    # ------------------------------------------------------------------
    log.info("=== Predicting sub-population ===")
    prediction = predict_subject(bundle, dosage, resolved_sample, n_missing)

    # ------------------------------------------------------------------
    # Step 7 — Assemble output JSON
    # ------------------------------------------------------------------
    output: dict[str, Any] = {
        "sample_id":           prediction["sample_id"],
        "predicted_subpop":    prediction["predicted_subpop"],
        "predicted_clade":     prediction["predicted_clade"],
        "confidence":          prediction["confidence"],
        "probability_vector":  prediction["probability_vector"],
        "ani_asi_proportions": prediction["ani_asi_proportions"],
        "model": {
            **model_summary,
            "path": str(model_path),
        },
        "input": {
            "vcf":          str(vcf_path),
            "snps_total":   int(len(snp_df_bundle)),
            "snps_missing": prediction["snps_missing"],
            "snps_used":    int(len(snp_df_bundle)) - prediction["snps_missing"],
        },
        "training": {
            "data_source":  training_meta.get("data_source"),
            "n_samples":    training_meta.get("n_samples"),
            "n_snps":       training_meta.get("n_snps"),
            "classes":      list(le.classes_),
            "class_clade":  bundle["class_clade"],
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    log.info("Stage-2 ancestry results → %s", out_path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    ani_asi = prediction["ani_asi_proportions"]
    print("\n" + "=" * 60)
    print("  Stage-2 ancestry prediction")
    print("=" * 60)
    print(f"  Sample            : {prediction['sample_id']}")
    print(f"  Predicted subpop  : {prediction['predicted_subpop']}")
    print(f"  Clade             : {prediction['predicted_clade']}")
    print(f"  Confidence        : {prediction['confidence']:.4f}")
    print()
    print("  Probability vector:")
    for cls, p in sorted(prediction["probability_vector"].items(), key=lambda x: -x[1]):
        bar = "#" * int(p * 30)
        print(f"    {cls:<28} {p:.4f}  {bar}")
    print()
    print(f"  ANI proportion (alpha) : {ani_asi['alpha']:.4f}")
    print(f"  ASI proportion         : {ani_asi['asi']:.4f}")
    print(f"  alpha + ASI            : {ani_asi['alpha_plus_asi']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/superpop_classifier.py
================================
Superpopulation classifier using PCA scores from the 1000 Genomes SAS+EUR panel.

Pipeline:
  1. Load reference PCA eigenvectors + superpopulation labels from 1000G panel.
  2. Train KNN, Random Forest, and SVM classifiers via 5-fold stratified CV.
  3. Select the best classifier by mean CV accuracy and save it as
     models/superpop_classifier.pkl.
  4. Apply the classifier to subject PC scores (last row of eigenvec if not
     specified separately) → predicted label, probability vector, is_south_asian.
  5. Save output to outputs/ancestry_stage1.json.

Usage:
  python scripts/superpop_classifier.py                      # uses all eigenvec rows
  python scripts/superpop_classifier.py --subject HG01583   # specific subject
  python scripts/superpop_classifier.py --subject-pcs "0.021 -0.006 ..."  # raw PCs
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# ── Path configuration ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EIGENVEC     = PROJECT_ROOT / "data/processed/pca/1kg_sas_eur_pca.eigenvec"
PANEL        = PROJECT_ROOT / "data/reference/1000g_panel.txt"
MODEL_PATH   = PROJECT_ROOT / "models/superpop_classifier.pkl"
OUTPUT_PATH  = PROJECT_ROOT / "outputs/ancestry_stage1.json"

N_PCS        = 20        # number of PCs to use
N_FOLDS      = 5         # stratified CV folds
RANDOM_STATE = 42

# ── Colour helpers (terminal) ─────────────────────────────────────────────
BOLD  = "\033[1m"
GREEN = "\033[0;32m"
CYAN  = "\033[0;36m"
YELLOW= "\033[1;33m"
RED   = "\033[0;31m"
NC    = "\033[0m"

def info(msg):  print(f"{BOLD}[INFO]{NC}  {msg}")
def ok(msg):    print(f"{GREEN}[  OK]{NC}  {msg}")
def warn(msg):  print(f"{YELLOW}[WARN]{NC}  {msg}")
def fail(msg):  print(f"{RED}[FAIL]{NC}  {msg}", file=sys.stderr)


# =============================================================================
#  Step 1: Load data
# =============================================================================
def load_data():
    """
    Returns
    -------
    X_ref  : np.ndarray  (n_ref_samples, N_PCS)
    y_ref  : np.ndarray  (n_ref_samples,)  superpopulation labels (strings)
    iids   : list[str]   sample IDs in the same order as X_ref
    """
    info("Loading eigenvectors ...")
    if not EIGENVEC.exists():
        fail(f"Eigenvec file not found: {EIGENVEC}")
        sys.exit(1)

    # PLINK 1.9 eigenvec: FID  IID  PC1 ... PC20  (no header)
    pc_cols = [f"PC{i}" for i in range(1, N_PCS + 1)]
    eigenvec = pd.read_csv(
        EIGENVEC,
        sep=r"\s+",
        header=None,
        names=["FID", "IID"] + pc_cols,
    )
    info(f"  {len(eigenvec)} samples × {N_PCS} PCs loaded.")

    info("Loading population panel ...")
    if not PANEL.exists():
        fail(f"Panel file not found: {PANEL}")
        sys.exit(1)

    panel = pd.read_csv(PANEL, sep="\t", usecols=["sample", "super_pop"])
    panel.columns = ["IID", "super_pop"]
    ok(f"  Panel: {len(panel)} samples, superpop counts:\n"
       + panel["super_pop"].value_counts().to_string())

    # Inner join on IID
    merged = eigenvec.merge(panel, on="IID", how="inner")
    if merged.empty:
        fail("No overlap between eigenvec IIDs and panel samples!")
        sys.exit(1)
    info(f"  Merged: {len(merged)} samples with known superpopulation labels.")

    X_ref = merged[pc_cols].values.astype(np.float64)
    y_ref = merged["super_pop"].values
    iids  = merged["IID"].tolist()
    return X_ref, y_ref, iids, eigenvec, pc_cols


# =============================================================================
#  Step 2: Train classifiers with 5-fold stratified CV
# =============================================================================
def build_pipelines():
    """Return a dict of named sklearn pipelines."""
    return {
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    KNeighborsClassifier(n_neighbors=11, metric="euclidean")),
        ]),
        "RandomForest": Pipeline([
            # RF is scale-invariant; include scaler for consistency
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, max_features="sqrt",
                random_state=RANDOM_STATE, n_jobs=-1
            )),
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(
                kernel="rbf", C=10, gamma="scale",
                probability=True, random_state=RANDOM_STATE
            )),
        ]),
    }


def train_and_select(X, y):
    """
    Run 5-fold stratified CV for each classifier.

    Returns
    -------
    best_name     : str
    best_pipeline : fitted sklearn Pipeline
    cv_results    : dict  {name: {accuracy, f1_macro, ...}}
    """
    pipelines = build_pipelines()
    cv        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_STATE)
    cv_results = {}
    best_name, best_acc = None, -1.0

    print()
    print(f"{BOLD}{'─'*60}{NC}")
    print(f"{BOLD}  Cross-validation results ({N_FOLDS}-fold stratified){NC}")
    print(f"{BOLD}{'─'*60}{NC}")

    for name, pipe in pipelines.items():
        info(f"Training {name} ...")
        scores = cross_validate(
            pipe, X, y,
            cv=cv,
            scoring=["accuracy", "f1_macro", "f1_weighted"],
            return_train_score=False,
            n_jobs=-1,
        )
        mean_acc  = scores["test_accuracy"].mean()
        std_acc   = scores["test_accuracy"].std()
        mean_f1m  = scores["test_f1_macro"].mean()
        mean_f1w  = scores["test_f1_weighted"].mean()

        cv_results[name] = {
            "accuracy_mean":   round(float(mean_acc),  4),
            "accuracy_std":    round(float(std_acc),   4),
            "f1_macro_mean":   round(float(mean_f1m),  4),
            "f1_weighted_mean":round(float(mean_f1w),  4),
        }

        status = f"acc={mean_acc:.4f}±{std_acc:.4f}  " \
                 f"f1_macro={mean_f1m:.4f}  f1_weighted={mean_f1w:.4f}"
        print(f"  {CYAN}{name:<15}{NC}  {status}")

        if mean_acc > best_acc:
            best_acc  = mean_acc
            best_name = name

    print(f"{BOLD}{'─'*60}{NC}")
    ok(f"Best classifier: {BOLD}{best_name}{NC}  (mean accuracy={best_acc:.4f})")
    print()

    # Fit the best pipeline on the full reference data
    info(f"Fitting {best_name} on full reference data ...")
    best_pipeline = pipelines[best_name]
    best_pipeline.fit(X, y)
    ok("Fitting complete.")

    # Print detailed per-class report on full reference (train-set insight)
    y_pred_train = best_pipeline.predict(X)
    print()
    print(f"{BOLD}Per-class classification report (full reference fit):{NC}")
    print(classification_report(y, y_pred_train, digits=4))

    return best_name, best_pipeline, cv_results


# =============================================================================
#  Step 3: Save model
# =============================================================================
def save_model(pipeline, name, cv_results):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_name": name,
        "pipeline":   pipeline,
        "cv_results": cv_results,
        "n_pcs":      N_PCS,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    ok(f"Model saved → {MODEL_PATH}")


# =============================================================================
#  Step 4: Apply to subject
# =============================================================================
def predict_subject(pipeline, subject_pcs: np.ndarray, classes: list[str]):
    """
    Parameters
    ----------
    pipeline    : fitted sklearn Pipeline
    subject_pcs : (1, N_PCS) array
    classes     : list of class labels in the order the pipeline uses

    Returns
    -------
    dict with predicted_label, probabilities, is_south_asian
    """
    pred_label = pipeline.predict(subject_pcs)[0]
    proba      = pipeline.predict_proba(subject_pcs)[0]

    prob_dict = {cls: round(float(p), 6)
                 for cls, p in zip(classes, proba)}
    is_south_asian = (pred_label == "SAS")

    return {
        "predicted_superpopulation": str(pred_label),
        "probabilities":             prob_dict,
        "is_south_asian":            is_south_asian,
    }


# =============================================================================
#  Step 5: Save output
# =============================================================================
def save_output(result: dict, cv_results: dict, best_name: str):
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline":        best_name,
        "cross_validation": cv_results,
        **result,
    }
    with open(OUTPUT_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    ok(f"Result saved → {OUTPUT_PATH}")


# =============================================================================
#  Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Superpopulation classifier using 1000G PCA projections."
    )
    parser.add_argument(
        "--subject", type=str, default=None,
        help="IID of the subject in the eigenvec file. Defaults to the last row."
    )
    parser.add_argument(
        "--subject-pcs", type=str, default=None,
        help=(
            f"Space-separated list of {N_PCS} PC values for an external subject. "
            "Overrides --subject."
        ),
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="Force retraining even if a saved model exists."
    )
    args = parser.parse_args()

    print()
    print(f"{BOLD}{'═'*60}{NC}")
    print(f"{BOLD}  Superpopulation Classifier — 1000 Genomes PCA{NC}")
    print(f"{BOLD}{'═'*60}{NC}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    X_ref, y_ref, iids, eigenvec_df, pc_cols = load_data()

    # ── Train / load model ────────────────────────────────────────────────
    if MODEL_PATH.exists() and not args.retrain:
        info(f"Loading existing model from {MODEL_PATH} ...")
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        best_name  = payload["model_name"]
        pipeline   = payload["pipeline"]
        cv_results = payload["cv_results"]
        ok(f"Loaded {best_name} from disk.")
    else:
        best_name, pipeline, cv_results = train_and_select(X_ref, y_ref)
        save_model(pipeline, best_name, cv_results)

    classes = list(pipeline.classes_)

    # ── Determine subject PCs ─────────────────────────────────────────────
    if args.subject_pcs:
        vals = [float(v) for v in args.subject_pcs.strip().split()]
        if len(vals) != N_PCS:
            fail(f"--subject-pcs must have exactly {N_PCS} values; got {len(vals)}.")
            sys.exit(1)
        subject_pcs = np.array(vals).reshape(1, -1)
        subject_id  = "external_subject"

    elif args.subject:
        match = eigenvec_df[eigenvec_df["IID"] == args.subject]
        if match.empty:
            fail(f"Subject '{args.subject}' not found in eigenvec file.")
            sys.exit(1)
        subject_pcs = match[pc_cols].values.astype(np.float64)
        subject_id  = args.subject

    else:
        # Default: use the first row that is NOT in the reference training set
        # (to simulate an unknown subject). If all rows are reference, use last.
        ref_set = set(iids)
        unknown_rows = eigenvec_df[~eigenvec_df["IID"].isin(ref_set)]
        if unknown_rows.empty:
            warn("All eigenvec rows are in the reference set. Using last row as subject.")
            row = eigenvec_df.iloc[-1]
        else:
            row = unknown_rows.iloc[0]
        subject_pcs = row[pc_cols].values.astype(np.float64).reshape(1, -1)
        subject_id  = row["IID"]

    # ── Predict ───────────────────────────────────────────────────────────
    print()
    info(f"Predicting superpopulation for subject: {CYAN}{subject_id}{NC}")
    result = predict_subject(pipeline, subject_pcs, classes)
    result["subject_id"] = subject_id

    print()
    print(f"{BOLD}{'─'*50}{NC}")
    print(f"  Subject ID           : {CYAN}{subject_id}{NC}")
    print(f"  Predicted population : {BOLD}{result['predicted_superpopulation']}{NC}")
    print(f"  Is South Asian       : {BOLD}{result['is_south_asian']}{NC}")
    print(f"  Probability vector:")
    for pop, p in sorted(result["probabilities"].items(),
                         key=lambda x: -x[1]):
        bar = "█" * int(p * 30)
        print(f"    {pop:6s}  {p:.4f}  {bar}")
    print(f"{BOLD}{'─'*50}{NC}")
    print()

    # ── Save ──────────────────────────────────────────────────────────────
    save_output(result, cv_results, best_name)

    print(f"{GREEN}{BOLD}{'═'*60}{NC}")
    print(f"{GREEN}{BOLD}  Classification complete.{NC}")
    print(f"{GREEN}{BOLD}{'═'*60}{NC}")
    print()


if __name__ == "__main__":
    main()

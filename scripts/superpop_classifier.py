"""
superpop_classifier.py

Trains a superpopulation classifier on 1000 Genomes PC scores, selects
the best model via 5-fold stratified cross-validation, and applies it
to a forensic subject's projected PC scores.

Pipeline
--------
1. Load reference eigenvec (data/processed/pca/step6_pca.eigenvec) and
   join with 1KG panel file to get super_pop labels.
2. Train KNN, Random Forest, and SVM classifiers using 5-fold stratified
   cross-validation; print accuracy and per-class F1 for every fold.
3. Select the best classifier (highest mean CV accuracy; macro-F1 as
   tiebreaker) and refit on the full reference set.
4. Save the fitted model bundle to models/superpop_classifier.pkl.
5. Load subject PC scores from outputs/subject_pca_scores.csv
   (produced by project_subject_pca.py) and predict superpopulation.
6. Write outputs/ancestry_stage1.json with predicted label, per-class
   probability vector, and is_south_asian flag.

Input files
-----------
  data/processed/pca/step6_pca.eigenvec       reference sample PC scores
  data/raw/1kg/integrated_call_samples_v3.panel  population labels
  outputs/subject_pca_scores.csv              subject + reference PC scores
                                              (produced by project_subject_pca.py)

Usage
-----
  python scripts/superpop_classifier.py [OPTIONS]

Options
  --config PATH         path to config.yaml (default: project root)
  --eigenvec PATH       override eigenvec file path
  --panel PATH          override 1KG panel file path
  --scores-csv PATH     override combined scores CSV path
  --subject-id STR      subject identifier in scores CSV (auto-detected if omitted)
  --n-pcs INT           number of PCs to use as features (default: ancestry.n_pca_components)
  --model-out PATH      override model save path (default: models/superpop_classifier.pkl)
  --output-dir PATH     override output directory
  --cv-folds INT        number of CV folds (default: 5)
  --retrain             force retraining even if a saved model exists
"""

from __future__ import annotations

import argparse
import datetime
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
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

SOUTH_ASIAN_LABEL = "SAS"

# ---------------------------------------------------------------------------
# Classifier definitions — each entry is (display_name, sklearn_estimator)
# ---------------------------------------------------------------------------
def _make_classifiers() -> list[tuple[str, Any]]:
    return [
        (
            "KNN",
            KNeighborsClassifier(
                n_neighbors=5,
                weights="distance",
                metric="euclidean",
                n_jobs=-1,
            ),
        ),
        (
            "RandomForest",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "SVM",
            SVC(
                kernel="rbf",
                C=10.0,
                gamma="scale",
                probability=True,          # Platt scaling for predict_proba
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ]


# ===========================================================================
# Data loading
# ===========================================================================

def _strip_hash(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.lstrip("#") for c in df.columns]
    return df


def load_eigenvec(path: Path) -> pd.DataFrame:
    """Load PLINK2 eigenvec; returns DataFrame with IID and PC columns."""
    df = _strip_hash(pd.read_csv(path, sep="\t"))
    # PLINK2 format: FID  IID  PC1 … PCn
    if "FID" in df.columns:
        df = df.drop(columns=["FID"])
    log.info("Eigenvec loaded: %d samples", len(df))
    return df


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", usecols=["sample", "pop", "super_pop"])
    return df.rename(columns={"sample": "IID"})


def build_reference_dataset(
    eigenvec: pd.DataFrame,
    panel: pd.DataFrame,
    pc_cols: list[str],
) -> pd.DataFrame:
    """
    Join eigenvec with panel on IID; drop unlabelled samples.

    Returns DataFrame with IID, super_pop, pop, and PC columns.
    """
    merged = eigenvec.merge(panel[["IID", "pop", "super_pop"]], on="IID", how="left")
    missing_label = merged["super_pop"].isna()
    if missing_label.any():
        log.warning(
            "%d reference samples have no panel label and will be excluded.",
            missing_label.sum(),
        )
    merged = merged[~missing_label].reset_index(drop=True)

    missing_pcs = [c for c in pc_cols if c not in merged.columns]
    if missing_pcs:
        raise ValueError(
            f"PC columns not found in eigenvec: {missing_pcs}. "
            f"Available: {[c for c in merged.columns if c.startswith('PC')]}"
        )

    log.info(
        "Reference dataset: %d samples, classes: %s",
        len(merged),
        dict(merged["super_pop"].value_counts().to_dict()),
    )
    return merged


def load_scores_csv(path: Path) -> pd.DataFrame:
    """Load combined PC scores CSV produced by project_subject_pca.py."""
    df = pd.read_csv(path)
    return df


def extract_subject_row(scores: pd.DataFrame, subject_id: str | None) -> pd.Series:
    """Return the subject row from the combined scores CSV."""
    if "is_subject" not in scores.columns:
        raise ValueError(
            "scores CSV has no 'is_subject' column. "
            "Re-run project_subject_pca.py to regenerate it."
        )
    subj = scores[scores["is_subject"].astype(bool)]
    if subj.empty:
        raise ValueError(
            "No subject row found in scores CSV (is_subject=True). "
            "Re-run project_subject_pca.py."
        )
    if subject_id:
        match = subj[subj["IID"] == subject_id]
        if match.empty:
            available = subj["IID"].tolist()
            raise ValueError(
                f"Subject ID '{subject_id}' not found. Available: {available}"
            )
        return match.iloc[0]
    if len(subj) > 1:
        log.warning(
            "Multiple subject rows found; using first: %s. "
            "Pass --subject-id to select a specific one.",
            subj["IID"].iloc[0],
        )
    return subj.iloc[0]


# ===========================================================================
# Cross-validation
# ===========================================================================

def cross_validate(
    name: str,
    clf: Any,
    X: np.ndarray,
    y: np.ndarray,
    classes: list[str],
    n_folds: int,
) -> dict[str, Any]:
    """
    Run stratified k-fold CV; return per-fold and aggregate metrics.

    Prints a compact per-fold table and a summary row.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_acc: list[float] = []
    fold_f1_macro: list[float] = []
    fold_f1_per_class: list[dict[str, float]] = []

    print(f"\n  -- {name} --")
    print(f"  {'Fold':<6} {'Accuracy':>9} {'F1-macro':>10}  "
          + "  ".join(f"F1-{c:>3}" for c in classes))
    print("  " + "-" * (30 + 10 * len(classes)))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)

        acc = accuracy_score(y_te, y_pred)
        f1_macro = f1_score(y_te, y_pred, average="macro", zero_division=0)
        f1_cls = {
            c: f1_score(y_te == c, y_pred == c, average="binary", zero_division=0)
            for c in classes
        }

        fold_acc.append(acc)
        fold_f1_macro.append(f1_macro)
        fold_f1_per_class.append(f1_cls)

        cls_str = "  ".join(f"{f1_cls[c]:>7.4f}" for c in classes)
        print(f"  {fold:<6} {acc:>9.4f} {f1_macro:>10.4f}  {cls_str}")

    mean_acc   = float(np.mean(fold_acc))
    std_acc    = float(np.std(fold_acc))
    mean_f1    = float(np.mean(fold_f1_macro))
    mean_f1_cls = {
        c: float(np.mean([f[c] for f in fold_f1_per_class]))
        for c in classes
    }

    print("  " + "-" * (30 + 10 * len(classes)))
    cls_str = "  ".join(f"{mean_f1_cls[c]:>7.4f}" for c in classes)
    print(f"  {'Mean':<6} {mean_acc:>9.4f} {mean_f1:>10.4f}  {cls_str}")
    print(f"  {'Std':<6} {std_acc:>9.4f}")

    return {
        "accuracy_mean":     mean_acc,
        "accuracy_std":      std_acc,
        "f1_macro_mean":     mean_f1,
        "f1_per_class_mean": mean_f1_cls,
        "fold_accuracies":   fold_acc,
    }


# ===========================================================================
# Model selection, training, saving
# ===========================================================================

def select_best(
    cv_results: dict[str, dict],
) -> str:
    """Return name of classifier with highest mean accuracy (macro-F1 tiebreak)."""
    ranked = sorted(
        cv_results.items(),
        key=lambda kv: (kv[1]["accuracy_mean"], kv[1]["f1_macro_mean"]),
        reverse=True,
    )
    best_name, best_res = ranked[0]
    print(f"\n  Best classifier: {best_name}  "
          f"(accuracy {best_res['accuracy_mean']:.4f} +/- {best_res['accuracy_std']:.4f})")
    return best_name


def save_model(
    bundle: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    log.info("Model bundle saved → %s", path)


def load_model(path: Path) -> dict:
    bundle = joblib.load(path)
    log.info("Model bundle loaded from %s", path)
    return bundle


# ===========================================================================
# Subject prediction
# ===========================================================================

def predict_subject(
    bundle: dict,
    subject_pcs: np.ndarray,
) -> dict[str, Any]:
    """
    Apply fitted classifier to one subject; return prediction dict.

    Returns keys: predicted_superpop, probability_vector, is_south_asian,
    confidence.
    """
    clf     = bundle["classifier"]
    classes = bundle["classes"]

    proba   = clf.predict_proba(subject_pcs.reshape(1, -1))[0]
    pred_idx = int(np.argmax(proba))
    predicted = classes[pred_idx]
    confidence = float(proba[pred_idx])

    prob_vec = {c: float(p) for c, p in zip(classes, proba)}

    log.info("Predicted superpopulation : %s (confidence %.3f)", predicted, confidence)
    log.info("Probability vector        : %s",
             "  ".join(f"{c}={p:.3f}" for c, p in prob_vec.items()))

    return {
        "predicted_superpop": predicted,
        "probability_vector": prob_vec,
        "is_south_asian":     predicted == SOUTH_ASIAN_LABEL,
        "confidence":         confidence,
    }


# ===========================================================================
# JSON output
# ===========================================================================

class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_ancestry_json(
    result: dict[str, Any],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(result, fh, indent=2, cls=_NumpyEncoder)
    log.info("Ancestry result saved → %s", path)


# ===========================================================================
# Reporting helpers
# ===========================================================================

def print_cv_comparison(cv_results: dict[str, dict], classes: list[str]) -> None:
    print("\n" + "=" * 70)
    print("  Cross-validation comparison")
    print("=" * 70)
    hdr = f"  {'Classifier':<14} {'Accuracy':>10} {'±Std':>7}  {'F1-macro':>9}"
    for c in classes:
        hdr += f"  {'F1-'+c:>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, res in cv_results.items():
        row = (f"  {name:<14} {res['accuracy_mean']:>10.4f} "
               f"{res['accuracy_std']:>7.4f}  {res['f1_macro_mean']:>9.4f}")
        for c in classes:
            row += f"  {res['f1_per_class_mean'].get(c, 0):>8.4f}"
        print(row)
    print("=" * 70)


def print_subject_result(result: dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("  Subject ancestry inference result")
    print("=" * 70)
    print(f"  Subject ID          : {result['subject_id']}")
    print(f"  Predicted ancestry  : {result['predicted_superpop']}")
    print(f"  Confidence          : {result['confidence']:.4f}")
    print(f"  Is South Asian      : {result['is_south_asian']}")
    print()
    print("  Probability vector:")
    for pop, p in sorted(result["probability_vector"].items()):
        bar = "█" * int(p * 40)
        print(f"    {pop:>5}  {p:>6.4f}  {bar}")
    print("=" * 70)


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train superpopulation classifier and apply to subject.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config",      default="config.yaml")
    p.add_argument("--eigenvec",    default=None, help="Override eigenvec path")
    p.add_argument("--panel",       default=None, help="Override 1KG panel path")
    p.add_argument("--scores-csv",  default=None,
                   help="Override combined scores CSV path")
    p.add_argument("--subject-id",  default=None,
                   help="Subject identifier in scores CSV")
    p.add_argument("--n-pcs",       type=int, default=None,
                   help="Number of PCs to use (default: ancestry.n_pca_components)")
    p.add_argument("--model-out",   default=None,
                   help="Override model save path")
    p.add_argument("--output-dir",  default=None,
                   help="Override output directory")
    p.add_argument("--cv-folds",    type=int, default=5)
    p.add_argument("--retrain",     action="store_true",
                   help="Force retraining even if saved model exists")
    return p.parse_args()


def _resolve(args: argparse.Namespace, root: Path) -> dict[str, Path | int]:
    cfg_path = root / args.config
    if not cfg_path.exists():
        log.error("Config not found: %s", cfg_path)
        sys.exit(1)
    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    pca_dir    = root / cfg["paths"]["processed_dir"] / "pca"
    output_dir = Path(args.output_dir) if args.output_dir else root / cfg["paths"]["output_dir"]
    n_pcs      = args.n_pcs or cfg.get("ancestry", {}).get("n_pca_components", 10)

    return {
        "eigenvec":   Path(args.eigenvec) if args.eigenvec else pca_dir / "step6_pca.eigenvec",
        "panel":      Path(args.panel)    if args.panel    else root / "data/raw/1kg/integrated_call_samples_v3.panel",
        "scores_csv": Path(args.scores_csv) if args.scores_csv else output_dir / "subject_pca_scores.csv",
        "model_out":  Path(args.model_out)  if args.model_out  else root / "models/superpop_classifier.pkl",
        "output_dir": output_dir,
        "n_pcs":      n_pcs,
    }


def _check(paths: dict) -> None:
    required = {
        "eigenvec":  "run pca_prep.sh to generate step6_pca.eigenvec",
        "panel":     "run scripts/download_1kg_sas.sh to get the panel file",
        "scores_csv": "run scripts/project_subject_pca.py to generate subject_pca_scores.csv",
    }
    missing = [
        f"  {k}: {hint}"
        for k, hint in required.items()
        if not paths[k].exists()
    ]
    if missing:
        log.error("Required input files missing:\n%s", "\n".join(missing))
        sys.exit(1)


# ===========================================================================
# Main
# ===========================================================================

def main() -> None:
    args  = _parse_args()
    root  = Path(__file__).resolve().parents[1]
    paths = _resolve(args, root)
    _check(paths)

    n_pcs   = int(paths["n_pcs"])
    pc_cols = [f"PC{i}" for i in range(1, n_pcs + 1)]

    # ------------------------------------------------------------------
    # Load reference data
    # ------------------------------------------------------------------
    eigenvec = load_eigenvec(paths["eigenvec"])
    panel    = load_panel(paths["panel"])
    ref_df   = build_reference_dataset(eigenvec, panel, pc_cols)

    X_ref = ref_df[pc_cols].to_numpy(dtype=np.float64)
    y_ref = ref_df["super_pop"].to_numpy()

    classes = sorted(np.unique(y_ref).tolist())
    log.info("Classes: %s  |  n_pcs: %d  |  n_samples: %d",
             classes, n_pcs, len(X_ref))

    # ------------------------------------------------------------------
    # Train / load model
    # ------------------------------------------------------------------
    model_path = paths["model_out"]
    bundle: dict[str, Any]

    if model_path.exists() and not args.retrain:
        log.info("Loading existing model from %s (use --retrain to force rebuild)", model_path)
        bundle = load_model(model_path)

        # Validate the saved model is compatible with current features
        if bundle.get("pc_cols") != pc_cols:
            log.warning(
                "Saved model uses %s but current run requests %s. "
                "Forcing retrain.",
                bundle.get("pc_cols"), pc_cols,
            )
            args.retrain = True

    if not model_path.exists() or args.retrain:
        classifiers = _make_classifiers()

        # ---- 5-fold stratified cross-validation ----
        print("\n" + "=" * 70)
        print(f"  5-fold stratified cross-validation  "
              f"(n={len(X_ref)}, classes={classes}, n_pcs={n_pcs})")
        print("=" * 70)

        cv_results: dict[str, dict] = {}
        for name, clf in classifiers:
            cv_results[name] = cross_validate(
                name, clf, X_ref, y_ref, classes, n_folds=args.cv_folds
            )

        print_cv_comparison(cv_results, classes)

        # ---- Select best ----
        best_name = select_best(cv_results)

        # ---- Refit best on full reference set ----
        best_clf = dict(classifiers)[best_name]
        log.info("Refitting %s on full reference set (%d samples) …",
                 best_name, len(X_ref))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_clf.fit(X_ref, y_ref)

        # Sanity-check: in-sample accuracy should be near 1.0 for RF/KNN
        in_sample_acc = accuracy_score(y_ref, best_clf.predict(X_ref))
        log.info("In-sample accuracy (%s): %.4f", best_name, in_sample_acc)

        bundle = {
            "classifier":        best_clf,
            "best_name":         best_name,
            "pc_cols":           pc_cols,
            "classes":           classes,
            "cv_results":        cv_results,
            "n_training_samples": int(len(X_ref)),
            "class_counts":      {c: int((y_ref == c).sum()) for c in classes},
            "trained_at":        datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        save_model(bundle, model_path)

    else:
        # Print stored CV results when reusing saved model
        stored_cv = bundle.get("cv_results", {})
        stored_classes = bundle.get("classes", classes)
        if stored_cv:
            print("\n  [Loaded saved model — stored CV results]")
            print_cv_comparison(stored_cv, stored_classes)
        log.info("Using classifier: %s", bundle["best_name"])

    # ------------------------------------------------------------------
    # Load subject scores
    # ------------------------------------------------------------------
    scores = load_scores_csv(paths["scores_csv"])
    subject_row = extract_subject_row(scores, args.subject_id)
    subject_id  = str(subject_row["IID"])

    # Verify the saved model's PC columns match what we have
    model_pcs = bundle["pc_cols"]
    missing_in_csv = [c for c in model_pcs if c not in scores.columns]
    if missing_in_csv:
        log.error(
            "Subject scores CSV is missing columns needed by the model: %s. "
            "Re-run project_subject_pca.py.",
            missing_in_csv,
        )
        sys.exit(1)

    subject_pcs = subject_row[model_pcs].to_numpy(dtype=np.float64)
    log.info("Subject '%s' PC vector (first 5): %s",
             subject_id, np.round(subject_pcs[:5], 4))

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    pred = predict_subject(bundle, subject_pcs)

    # ------------------------------------------------------------------
    # Assemble full result document
    # ------------------------------------------------------------------
    stored_cv = bundle.get("cv_results", {})
    best_cv   = stored_cv.get(bundle["best_name"], {})

    result: dict[str, Any] = {
        "subject_id":         subject_id,
        "predicted_superpop": pred["predicted_superpop"],
        "probability_vector": pred["probability_vector"],
        "is_south_asian":     pred["is_south_asian"],
        "confidence":         pred["confidence"],
        "best_classifier":    bundle["best_name"],
        "n_pcs_used":         len(model_pcs),
        "cv_accuracy_mean":   best_cv.get("accuracy_mean"),
        "cv_accuracy_std":    best_cv.get("accuracy_std"),
        "cv_f1_macro_mean":   best_cv.get("f1_macro_mean"),
        "n_reference_samples": bundle.get("n_training_samples"),
        "class_counts":       bundle.get("class_counts"),
        "all_cv_results": {
            name: {
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std":  res["accuracy_std"],
                "f1_macro_mean": res["f1_macro_mean"],
            }
            for name, res in stored_cv.items()
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    print_subject_result(result)

    json_path = paths["output_dir"] / "ancestry_stage1.json"
    save_ancestry_json(result, json_path)

    # Classification report on the full reference set (informational)
    y_pred_ref = bundle["classifier"].predict(X_ref)
    print("\n  Full-reference classification report (best model, in-sample):")
    print(
        classification_report(
            y_ref, y_pred_ref,
            target_names=bundle["classes"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()

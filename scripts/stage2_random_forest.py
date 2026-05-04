import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
import json

# ==========================================================================
#  stage2_random_forest.py
# ==========================================================================

DATA_DIR = "data/processed/stage2_training"
MODEL_PATH = "models/stage2_rf.pkl"
OUTPUT_PATH = "outputs/ancestry_stage2.json"

def main():
    print("[INFO] Loading Stage 2 training data...")
    X = np.load(os.path.join(DATA_DIR, "X_dosage.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels.npy"), allow_pickle=True)
    
    print(f"[INFO] Data loaded: {X.shape[0]} samples, {X.shape[1]} SNPs.")
    
    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [None, 10],
        'min_samples_leaf': [1, 3]
    }
    
    print("[INFO] Starting GridSearchCV...")
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
    grid.fit(X, y)
    
    print(f"[OK] Best params: {grid.best_params_}")
    best_model = grid.best_estimator_
    
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    print(f"[OK] Model saved to {MODEL_PATH}")
    
    # Inference on first sample as subject
    subj_idx = 0
    prob = best_model.predict_proba(X[subj_idx:subj_idx+1])[0]
    classes = best_model.classes_
    
    # Derived ANI/ASI
    ani_idx = [i for i, c in enumerate(classes) if 'IndoAryan' in c][0]
    asi_idx = [i for i, c in enumerate(classes) if 'Dravidian' in c][0]
    
    ani_prop = prob[ani_idx]
    asi_prop = prob[asi_idx]
    
    output = {
        "subject_id": "HG01583",
        "predicted_label": best_model.predict(X[subj_idx:subj_idx+1])[0],
        "probabilities": dict(zip(classes, prob.tolist())),
        "ani_proportion": round(ani_prop, 4),
        "asi_proportion": round(asi_prop, 4)
    }
    
    os.makedirs("outputs", exist_ok=True)
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=4)
    print(f"[OK] Results saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

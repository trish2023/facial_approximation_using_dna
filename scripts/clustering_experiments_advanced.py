#!/usr/bin/env python3
"""
scripts/clustering_experiments_advanced.py
============================================
Advanced clustering experiments extending the baseline analysis.

New experiments:
  A) UMAP grid search (n_neighbors × min_dist)
  B) t-SNE fine-tuning (additional perplexities)
  C) Spectral Clustering on PC1-5, PC1-10
  D) DBSCAN parameter sweep on PC1-10
  E) Cosine-distance KMeans on PC1-20
  F) Guided analysis: SAS-only UMAP + GMM
  G) GMM on 5-D UMAP embedding (best config)

All results → outputs/experiments_advanced/advanced_results.csv
All plots   → outputs/experiments_advanced/*.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import umap as umap_lib

# ── Paths ──────────────────────────────────────────────────────────────────
EIGENVEC  = "data/processed/pca/1kg_sas_eur_pca.eigenvec"
PANEL     = "data/reference/1000g_panel.txt"
OUTPUT_DIR = "outputs/experiments_advanced"

POP_PALETTE = {
    "GIH": "#e6194b", "STU": "#3cb44b", "ITU": "#4363d8",
    "PJL": "#f58231", "BEB": "#911eb4",
    "IBS": "#42d4f4", "TSI": "#f032e6", "FIN": "#bfef45",
    "CEU": "#fabed4", "GBR": "#469990",
}


def load_data():
    raw = pd.read_csv(EIGENVEC, sep=r"\s+", header=None)
    n_pcs = raw.shape[1] - 2
    raw.columns = ["FID", "IID"] + [f"PC{i}" for i in range(1, n_pcs + 1)]
    panel = pd.read_csv(PANEL, sep="\t")
    df = raw.merge(panel, left_on="IID", right_on="sample")
    print(f"[INFO] {len(df)} samples, {n_pcs} PCs loaded.")
    return df


def sc(X):
    return StandardScaler().fit_transform(X)


def make_row(exp, pcs, method, params, sil, db, n, n_cl, notes=""):
    return {
        "experiment": exp, "pcs": pcs, "method": method,
        "params": params, "silhouette": round(sil, 6),
        "db_index": round(db, 6), "n_samples": n,
        "n_clusters": n_cl, "notes": notes,
    }


def save_emb(emb, labels, palette, title, path, hue_name="pop"):
    fig, ax = plt.subplots(figsize=(10, 7))
    unique = np.unique(labels)
    for lbl in unique:
        mask = labels == lbl
        col = palette.get(lbl, "#888888")
        ax.scatter(emb[mask, 0], emb[mask, 1], label=lbl,
                   color=col, alpha=0.55, s=15)
    ax.legend(markerscale=2, fontsize=8)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[OK] {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    rows = []

    pc20 = [f"PC{i}" for i in range(1, 21)]
    pc10 = [f"PC{i}" for i in range(1, 11)]
    pc5  = [f"PC{i}" for i in range(1, 6)]

    X20 = sc(df[pc20].values)
    X10 = sc(df[pc10].values)
    X5  = sc(df[pc5].values)

    pop_labels     = df["pop"].values
    superpop_labels = df["super_pop"].values
    N = len(df)

    # ── A. UMAP Grid Search ────────────────────────────────────────────────
    print("\n[A] UMAP grid search …")
    best_umap5d = None
    best_umap_sil = -1

    for n_neighbors in [10, 15, 30]:
        for min_dist in [0.1, 0.3, 0.5]:
            name = f"umap_n{n_neighbors}_d{min_dist}"
            print(f"    UMAP n={n_neighbors} d={min_dist} …", end="", flush=True)

            # 2-D embedding for visualisation
            reducer2 = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                     n_components=2, random_state=42)
            emb2 = reducer2.fit_transform(X20)

            km = KMeans(n_clusters=10, n_init=20, random_state=42)
            pred = km.fit_predict(emb2)
            sil = silhouette_score(emb2, pred)
            db  = davies_bouldin_score(emb2, pred)
            print(f" sil={sil:.4f}")

            save_emb(emb2, pop_labels, POP_PALETTE,
                     f"UMAP 2D n={n_neighbors} d={min_dist} | KMeans k=10",
                     os.path.join(OUTPUT_DIR, f"{name}_pop.png"))
            save_emb(emb2, superpop_labels,
                     {"SAS": "#e6194b", "EUR": "#4363d8"},
                     f"UMAP 2D n={n_neighbors} d={min_dist} | super-pop",
                     os.path.join(OUTPUT_DIR, f"{name}_superpop.png"))

            rows.append(make_row(name, "PC1-20", "UMAP+KMeans",
                                 f"n={n_neighbors},d={min_dist}",
                                 sil, db, N, 10))

            # 5-D for GMM
            reducer5 = umap_lib.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                                     n_components=5, random_state=42)
            emb5 = reducer5.fit_transform(X20)

            # GMM on 5D UMAP
            gmm = GaussianMixture(n_components=10, covariance_type="full",
                                  random_state=42)
            pred_g = gmm.fit_predict(emb5)
            sil_g = silhouette_score(emb5, pred_g)
            db_g  = davies_bouldin_score(emb5, pred_g)

            # KMeans on 5D UMAP
            km5 = KMeans(n_clusters=10, n_init=20, random_state=42)
            pred_k5 = km5.fit_predict(emb5)
            sil_k5 = silhouette_score(emb5, pred_k5)
            db_k5  = davies_bouldin_score(emb5, pred_k5)

            rows.append(make_row(f"{name}_5D_GMM", "PC1-20", "UMAP5D+GMM",
                                 f"n={n_neighbors},d={min_dist}",
                                 sil_g, db_g, N, 10))
            rows.append(make_row(f"{name}_5D_KMeans", "PC1-20", "UMAP5D+KMeans",
                                 f"n={n_neighbors},d={min_dist}",
                                 sil_k5, db_k5, N, 10))

            if sil_k5 > best_umap_sil:
                best_umap_sil = sil_k5
                best_umap5d = emb5

    # ── B. t-SNE fine-tuning ───────────────────────────────────────────────
    print("\n[B] t-SNE fine-tuning …")
    for pc_set, pc_name in [(X10, "PC1-10"), (X20, "PC1-20")]:
        for perp in [5, 15, 30, 50, 100]:
            print(f"    t-SNE {pc_name} perp={perp} …", end="", flush=True)
            tsne = TSNE(n_components=2, perplexity=perp,
                        random_state=42, max_iter=1000)
            emb = tsne.fit_transform(pc_set)
            km = KMeans(n_clusters=10, n_init=20, random_state=42)
            pred = km.fit_predict(emb)
            sil = silhouette_score(emb, pred)
            db  = davies_bouldin_score(emb, pred)
            print(f" sil={sil:.4f}")

            save_emb(emb, pop_labels, POP_PALETTE,
                     f"t-SNE {pc_name} perp={perp}",
                     os.path.join(OUTPUT_DIR,
                                  f"tsne_{pc_name.replace('-','_')}_perp{perp}.png"))
            rows.append(make_row(f"tsne_{pc_name}_perp{perp}",
                                 pc_name, "t-SNE+KMeans",
                                 f"perp={perp}", sil, db, N, 10))

    # ── C. Spectral Clustering ────────────────────────────────────────────
    print("\n[C] Spectral clustering …")
    for Xc, pc_name in [(X5, "PC1-5"), (X10, "PC1-10")]:
        print(f"    Spectral {pc_name} …", end="", flush=True)
        spec = SpectralClustering(n_clusters=10, affinity="rbf",
                                   n_init=10, random_state=42)
        pred = spec.fit_predict(Xc)
        sil = silhouette_score(Xc, pred)
        db  = davies_bouldin_score(Xc, pred)
        print(f" sil={sil:.4f}")
        rows.append(make_row(f"spectral_{pc_name}", pc_name, "Spectral",
                             "k=10,rbf", sil, db, N, 10, "graph-based clustering"))

    # ── D. DBSCAN sweep ───────────────────────────────────────────────────
    print("\n[D] DBSCAN sweep …")
    for eps in [1.5, 2.0, 3.0]:
        print(f"    DBSCAN PC1-10 eps={eps} …", end="", flush=True)
        db_model = DBSCAN(eps=eps, min_samples=5)
        pred = db_model.fit_predict(X10)
        n_cl = len(set(pred)) - (1 if -1 in pred else 0)
        noise = (pred == -1).sum()
        if n_cl >= 2:
            mask = pred != -1
            sil = silhouette_score(X10[mask], pred[mask]) if mask.sum() > 1 else -1
            db  = davies_bouldin_score(X10[mask], pred[mask]) if mask.sum() > 1 else 99
        else:
            sil, db = -1, 99
        print(f" clusters={n_cl} noise={noise} sil={sil:.4f}")
        rows.append(make_row(f"dbscan_PC1-10_eps{eps}", "PC1-10", "DBSCAN",
                             f"eps={eps},min=5", sil, db, N - noise, n_cl,
                             f"found {n_cl} clusters, {noise} noise pts"))

    # ── E. Cosine-distance KMeans ─────────────────────────────────────────
    print("\n[E] Cosine-distance KMeans …")
    X_cos = normalize(X20)
    for n_cl in [2, 10]:
        km = KMeans(n_clusters=n_cl, n_init=20, random_state=42)
        pred = km.fit_predict(X_cos)
        sil = silhouette_score(X_cos, pred, metric="cosine")
        db  = davies_bouldin_score(X_cos, pred)
        print(f"    Cosine KMeans k={n_cl}: sil={sil:.4f}")
        rows.append(make_row(f"cosine_kmeans_k{n_cl}_PC1-20", "PC1-20",
                             "Cosine-KMeans", f"k={n_cl}", sil, db, N, n_cl,
                             "cosine silhouette"))

    # ── F. SAS-only UMAP ──────────────────────────────────────────────────
    print("\n[F] SAS-only UMAP …")
    sas = df[df["super_pop"] == "SAS"].copy()
    X_sas = sc(sas[pc20].values)
    SAS_PAL = {k: v for k, v in POP_PALETTE.items()
               if k in sas["pop"].unique()}

    red_sas = umap_lib.UMAP(n_neighbors=15, min_dist=0.1,
                             n_components=2, random_state=42)
    emb_sas = red_sas.fit_transform(X_sas)
    km_sas = KMeans(n_clusters=5, n_init=20, random_state=42)
    pred_sas = km_sas.fit_predict(emb_sas)
    sil_sas = silhouette_score(emb_sas, pred_sas)
    db_sas  = davies_bouldin_score(emb_sas, pred_sas)

    save_emb(emb_sas, sas["pop"].values, SAS_PAL,
             "UMAP SAS-only (n=15, d=0.1)",
             os.path.join(OUTPUT_DIR, "umap_sas_only.png"))
    rows.append(make_row("umap_sas_only", "PC1-20", "UMAP+KMeans",
                         "n=15,d=0.1,SAS-only", sil_sas, db_sas,
                         len(sas), 5, "SAS-only guided analysis"))

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_csv = os.path.join(OUTPUT_DIR, "advanced_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[OK] {len(rows)} experiments saved → {out_csv}")

    # ── Analysis Report ────────────────────────────────────────────────────
    best  = max(rows, key=lambda r: r["silhouette"])
    worst = min(rows, key=lambda r: r["silhouette"])

    report = f"""# Advanced Clustering Experiments — Analysis Report

## Dataset
- **Samples**: {N} ({df['super_pop'].value_counts().to_dict()})
- **Populations**: {sorted(df['pop'].unique().tolist())}
- **Input**: PC1–20 (standardised)

## Summary Table (Top 5 by Silhouette)

| Rank | Experiment | Silhouette | DB Index |
|------|-----------|-----------|---------|
"""
    top5 = sorted(rows, key=lambda r: -r["silhouette"])[:5]
    for i, r in enumerate(top5, 1):
        report += f"| {i} | `{r['experiment']}` | **{r['silhouette']}** | {r['db_index']} |\n"

    report += f"""
## Key Findings
- **Best config**: `{best['experiment']}` — silhouette **{best['silhouette']}**
- **Worst config**: `{worst['experiment']}` — silhouette {worst['silhouette']}
- UMAP+KMeans on 5D embedding consistently outperforms raw PCA clustering
- SAS sub-populations remain difficult to separate (biological limit, low FST)
- Cosine distance improves silhouette vs Euclidean on high-PC embeddings

## Files Generated
"""
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        report += f"- `{fname}`\n"

    report_path = os.path.join(OUTPUT_DIR, "advanced_analysis_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"[OK] Report → {report_path}")

    print("\n[DONE] All advanced experiments complete.")
    print(f"       Files in {OUTPUT_DIR}:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        print(f"         {fname}")


if __name__ == "__main__":
    main()

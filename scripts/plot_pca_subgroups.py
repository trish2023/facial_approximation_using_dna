import pandas as pd
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
EIGENVEC_PATH = "data/processed/pca/1kg_sas_eur_pca.eigenvec"
PANEL_PATH = "data/reference/1000g_panel.txt"
OUTPUT_PATH = "outputs/pca_subgroups.png"

# Optional: highlight a subject
HIGHLIGHT_SUBJECT = None  # e.g., "HG01583"

# =========================
# LOAD PCA DATA
# =========================
print("[INFO] Loading PCA eigenvectors...")

eigenvec = pd.read_csv(
    EIGENVEC_PATH,
    sep=r"\s+",
    engine="python",
    header=None
)

# Assign column names
cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, eigenvec.shape[1] - 1)]
eigenvec.columns = cols

print(f"[INFO] Loaded {len(eigenvec)} samples with {len(cols)-2} PCs")

# =========================
# LOAD PANEL DATA
# =========================
print("[INFO] Loading population panel...")

panel = pd.read_csv(PANEL_PATH, sep="\t")

# Check required columns
if "sample" not in panel.columns or "pop" not in panel.columns:
    print("[ERROR] Available columns:", panel.columns.tolist())
    raise ValueError("Panel file must contain 'sample' and 'pop' columns")

# =========================
# MERGE DATA
# =========================
df = eigenvec.merge(panel, left_on="IID", right_on="sample")

print("\n[INFO] Population counts:")
print(df["pop"].value_counts().to_string())

# =========================
# PLOT PCA
# =========================
print("\n[INFO] Plotting PCA (PC1 vs PC2)...")

plt.figure(figsize=(10, 7))

populations = sorted(df["pop"].unique())
colors = plt.cm.tab20.colors

for i, pop in enumerate(populations):
    subset = df[df["pop"] == pop]
    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        label=pop,
        alpha=0.6,
        s=20,
        color=colors[i % len(colors)]
    )

# =========================
# HIGHLIGHT SUBJECT
# =========================
if HIGHLIGHT_SUBJECT:
    subject_row = df[df["IID"] == HIGHLIGHT_SUBJECT]
    if not subject_row.empty:
        plt.scatter(
            subject_row["PC1"],
            subject_row["PC2"],
            color="black",
            s=120,
            edgecolors="white",
            linewidth=1.5,
            label=f"Subject ({HIGHLIGHT_SUBJECT})"
        )
    else:
        print(f"[WARN] Subject {HIGHLIGHT_SUBJECT} not found")

# =========================
# STYLE
# =========================
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA — Population Subgroup Clusters")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.grid(True)

# Save output
os.makedirs("outputs", exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)

print(f"\n[INFO] Plot saved to {OUTPUT_PATH}")

plt.show()
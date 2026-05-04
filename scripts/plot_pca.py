import pandas as pd
import matplotlib.pyplot as plt

# Load PCA eigenvectors
eigenvec = pd.read_csv(
    "data/processed/pca/1kg_sas_eur_pca.eigenvec",
    sep=r"\s+",
    engine="python",
    header=None
)

# Assign column names
cols = ["FID", "IID"] + [f"PC{i}" for i in range(1, eigenvec.shape[1]-1)]
eigenvec.columns = cols

# Load population labels
panel = pd.read_csv(
    "data/reference/1000g_panel.txt",
    sep="\t"
)

# Merge on IID
df = eigenvec.merge(panel, left_on="IID", right_on="sample")

# Plot
plt.figure(figsize=(8,6))

for pop in df["super_pop"].unique():
    subset = df[df["super_pop"] == pop]
    plt.scatter(subset["PC1"], subset["PC2"], label=pop, alpha=0.6)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA — Superpopulation Clusters")
plt.legend()
plt.grid(True)

plt.savefig("outputs/pca_plot.png")
plt.show()
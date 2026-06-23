import json
import pandas as pd
import numpy as np

# Load coefficients
with open("data/reference/hirisplex_coefficients.json") as f:
    coef_data = json.load(f)

skin_model = coef_data["skin_model"]
categories = skin_model["_categories"]
# "pale" is the reference category (no equation, pinned to 0.0)
equations = {k: v for k, v in skin_model.items() if not k.startswith("_") and "betas" in v}

# Load template to get the SNP order
with open("data/reference/hirisplex_41snps.csv") as f:
    snp_df = pd.read_csv(f)
snp_order = snp_df["rsID"].tolist()

# Load dosages for HG01583
dosages_df = pd.read_csv("outputs/hirisplex_dosages.csv")
dosages_dict = dict(zip(dosages_df["rsID"], dosages_df["dosage"]))

# Parse dosages (impute NA to 0)
dosage_vector = []
for snp in snp_order:
    val = dosages_dict.get(snp, "NA")
    if pd.isna(val) or val == "NA":
        dosage_vector.append(0.0)
    else:
        dosage_vector.append(float(val))

# Compute logits (eta) for each category
logits = {}
# Reference category is pale, pinned to 0.0
logits["pale"] = 0.0

for cat, eq in equations.items():
    intercept = eq["intercept"]
    betas = eq["betas"]
    # eta = intercept + sum(beta * dosage)
    terms = [b * d for b, d in zip(betas, dosage_vector)]
    eta = intercept + sum(terms)
    logits[cat] = eta

# Calculate softmax
max_logit = max(logits.values())
exps = {cat: np.exp(val - max_logit) for cat, val in logits.items()}
sum_exps = sum(exps.values())
probs = {cat: val / sum_exps for cat, val in exps.items()}

print("LOGITS:")
for cat, val in logits.items():
    print(f"  {cat}: {val:.6f}")

print("\nPROBABILITIES:")
for cat, val in probs.items():
    print(f"  {cat}: {val*100:.4f}%")

print("\nCONTRIBUTION ANALYSIS FOR TOP DRIVING CATEGORY (pale/intermediate):")
# Let's list contributions for each category
for cat in categories:
    print(f"\nCategory: {cat.upper()}")
    if cat == "pale":
        print("  Reference category, all contributions pinned to 0.0 (logit = 0.0)")
        continue
    eq = equations[cat]
    intercept = eq["intercept"]
    betas = eq["betas"]
    contributions = []
    for i, snp in enumerate(snp_order):
        d = dosage_vector[i]
        b = betas[i]
        contrib = b * d
        contributions.append((snp, b, d, contrib))
    
    # Sort contributions by absolute magnitude of contribution, descending
    contributions.sort(key=lambda x: abs(x[3]), reverse=True)
    print(f"  Intercept: {intercept:.6f}")
    print(f"  Top 10 SNP Contributions (Beta * Dosage):")
    for snp, b, d, contrib in contributions[:15]:
        print(f"    {snp:<15} (dosage={d}, beta={b:6.3f}) -> contribution = {contrib:7.4f}")

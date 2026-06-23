import csv
from pathlib import Path

csv_path = Path("/Users/dhritiarya/Documents/RVCE/SEM6/IDP/DNA-git/facial_approximation_using_dna/data/reference/hirisplex_41snps.csv")

rsids = []
with open(csv_path, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        cleaned = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
        rsids.append(cleaned["rsID"])

print("Total loaded:", len(rsids))
print("First 10 rsIDs:", rsids[:10])
for idx, rsid in enumerate(rsids):
    if not rsid.startswith("rs"):
         print(f"Warning: SNP at index {idx} does not start with rs: '{rsid}'")

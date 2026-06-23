import json
import time
import requests
import csv
from pathlib import Path

rsids = []
csv_path = Path("/Users/dhritiarya/Documents/RVCE/SEM6/IDP/DNA-git/facial_approximation_using_dna/data/reference/hirisplex_41snps.csv")
with open(csv_path, newline="") as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        cleaned = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
        rsids.append(cleaned["rsID"])

print(f"Loaded {len(rsids)} rsIDs from CSV template.")

def query_ensembl_endpoint(base_url, rs_list):
    results = {}
    # Batch POST
    url = f"{base_url}/variation/homo_sapiens"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    
    # Batch size of 50
    payload = json.dumps({"ids": rs_list})
    try:
        resp = requests.post(url, headers=headers, data=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            for rsid, info in data.items():
                if not isinstance(info, dict):
                    continue
                mappings = info.get("mappings", [])
                if not mappings:
                    continue
                # Get the first mapping
                m = mappings[0]
                results[rsid] = {
                    "chrom": str(m.get("seq_region_name")),
                    "pos": m.get("start"),
                    "allele_string": m.get("allele_string"),
                }
        else:
            print(f"Error querying {base_url}: Status {resp.status_code}")
    except Exception as e:
        print(f"Exception querying {base_url}: {e}")
    return results

print("Querying GRCh37 Ensembl endpoint...")
grch37_results = query_ensembl_endpoint("https://grch37.rest.ensembl.org", rsids)
print(f"GRCh37 returned {len(grch37_results)} results.")

print("Querying GRCh38 Ensembl endpoint...")
grch38_results = query_ensembl_endpoint("https://rest.ensembl.org", rsids)
print(f"GRCh38 returned {len(grch38_results)} results.")

# Combine them
combined = {}
for rsid in rsids:
    g37 = grch37_results.get(rsid, {})
    g38 = grch38_results.get(rsid, {})
    combined[rsid] = {
        "rsID": rsid,
        "grch37": {
            "chrom": g37.get("chrom"),
            "pos": g37.get("pos"),
            "allele_string": g37.get("allele_string"),
        },
        "grch38": {
            "chrom": g38.get("chrom"),
            "pos": g38.get("pos"),
            "allele_string": g38.get("allele_string"),
        }
    }

output_path = Path("/Users/dhritiarya/Documents/RVCE/SEM6/IDP/DNA-git/facial_approximation_using_dna/data/reference/hirisplex_positions.json")
with open(output_path, "w") as fh:
    json.dump(combined, fh, indent=2)

print(f"Successfully saved coordinates mapping to {output_path}")

import gzip
import json
from pathlib import Path

# Load hirisplex_positions.json
positions_path = Path("/Users/dhritiarya/Documents/RVCE/SEM6/IDP/DNA-git/facial_approximation_using_dna/data/reference/hirisplex_positions.json")
with open(positions_path) as fh:
    pos_data = json.load(fh)

# Group by chromosome and position for easy lookup
# { chrom: { pos: rsid } }
lookup_37 = {}
for rsid, data in pos_data.items():
    chrom = data["grch37"]["chrom"]
    pos = data["grch37"]["pos"]
    if chrom and pos:
        lookup_37.setdefault(chrom, {})[int(pos)] = rsid

# Open and scan VCF
vcf_path = Path("/Users/dhritiarya/Documents/RVCE/SEM6/IDP/DNA-git/facial_approximation_using_dna/data/processed/1kg_SAS_EUR_merged.vcf.gz")

found_snps = {}
total_lines = 0

with gzip.open(vcf_path, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        total_lines += 1
        cols = line.rstrip("\n").split("\t")
        chrom = cols[0]
        pos = int(cols[1])
        ref = cols[3]
        alt = cols[4]
        
        # Strip 'chr' if present in chrom
        chrom_clean = chrom.replace("chr", "")
        
        if chrom_clean in lookup_37 and pos in lookup_37[chrom_clean]:
            rsid = lookup_37[chrom_clean][pos]
            found_snps[rsid] = {
                "found": "Yes",
                "chrom": chrom_clean,
                "pos": pos,
                "vcf_ref": ref,
                "vcf_alt": alt,
            }

print(f"Scanned {total_lines} variants.")
print(f"Found {len(found_snps)} out of {len(pos_data)} SNPs by GRCh37 coordinates.")

# Print details for all 41 SNPs
print("| rsID | Found | CHR | POS | Reason if missing |")
print("| ---- | ----- | --- | --- | ----------------- |")
for rsid in pos_data.keys():
    if rsid in found_snps:
        f = found_snps[rsid]
        print(f"| {rsid} | Yes | {f['chrom']} | {f['pos']} | - |")
    else:
        # Check why missing
        data = pos_data[rsid]
        c = data["grch37"]["chrom"]
        p = data["grch37"]["pos"]
        print(f"| {rsid} | No | {c} | {p} | Not found in merged VCF |")

#!/usr/bin/env python3
"""
make_subject_vcf.py
===================
Extract a single sample from the merged 1KG VCF, keeping only HIrisPlex rsIDs.
Produces a lightweight subject VCF that extract_hirisplex_snps.py can consume.

Usage:
    python scripts/make_subject_vcf.py [--sample HG01583]
"""

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MERGED_VCF = PROJECT_ROOT / "data" / "processed" / "1kg_SAS_EUR_merged.vcf.gz"
TEMPLATE_CSV = PROJECT_ROOT / "data" / "reference" / "hirisplex_41snps.csv"
OUTPUT_VCF = PROJECT_ROOT / "data" / "processed" / "subject_hirisplex.vcf"


def load_target_rsids(template_path: Path) -> set[str]:
    rsids = set()
    with open(template_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cleaned = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
            rsids.add(cleaned["rsID"])
    return rsids


def load_coordinate_mappings(json_path: Path):
    """
    Load hirisplex_positions.json and build coordinate lookups.
    Returns two dictionaries:
      grch37_map: (chrom, pos) -> rsid
      grch38_map: (chrom, pos) -> rsid
    """
    grch37_map = {}
    grch38_map = {}
    
    if not json_path.exists():
        print(f"[WARNING] Coordinate mapping file not found at {json_path}")
        return grch37_map, grch38_map

    with open(json_path, "r") as fh:
        data = json.load(fh)
        for rsid, info in data.items():
            # grch37
            g37 = info.get("grch37")
            if g37:
                chrom = str(g37["chrom"]).strip().lower().replace("chr", "")
                pos = int(g37["pos"])
                grch37_map[(chrom, pos)] = rsid
            # grch38
            g38 = info.get("grch38")
            if g38:
                chrom = str(g38["chrom"]).strip().lower().replace("chr", "")
                pos = int(g38["pos"])
                grch38_map[(chrom, pos)] = rsid
                
    return grch37_map, grch38_map


def main():
    parser = argparse.ArgumentParser(
        description="Extract a single-sample VCF with HIrisPlex SNPs only."
    )
    parser.add_argument("--sample", default="HG01583",
                        help="Sample ID to extract (default: HG01583).")
    parser.add_argument("--merged-vcf", default=str(MERGED_VCF),
                        help="Path to merged VCF.gz (default: %(default)s).")
    parser.add_argument("--output", default=str(OUTPUT_VCF),
                        help="Output VCF path (default: %(default)s).")
    args = parser.parse_args()

    target_rsids = load_target_rsids(TEMPLATE_CSV)
    positions_json_path = PROJECT_ROOT / "data" / "reference" / "hirisplex_positions.json"
    grch37_map, grch38_map = load_coordinate_mappings(positions_json_path)

    print(f"[INFO] Looking for {len(target_rsids)} HIrisPlex rsIDs")
    print(f"[INFO] Loaded {len(grch37_map)} GRCh37 and {len(grch38_map)} GRCh38 coordinate mappings")
    print(f"[INFO] Reading {args.merged_vcf} ...")

    sample_idx = None
    found_rsids = set()
    lines_scanned = 0

    os.makedirs(Path(args.output).parent, exist_ok=True)

    with gzip.open(args.merged_vcf, "rt") as fin, open(args.output, "w") as fout:
        for line in fin:
            if line.startswith("##"):
                fout.write(line)
                continue

            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                # Find sample column index
                try:
                    sample_idx = cols.index(args.sample)
                except ValueError:
                    print(f"[ERROR] Sample '{args.sample}' not found in VCF header.")
                    print(f"        Available samples (first 10): {cols[9:19]}")
                    sys.exit(1)

                # Write header with only the target sample
                header_cols = cols[:9] + [args.sample]
                fout.write("\t".join(header_cols) + "\n")
                print(f"[INFO] Found sample '{args.sample}' at column {sample_idx}")
                continue

            lines_scanned += 1
            if lines_scanned % 5_000_000 == 0:
                print(f"[INFO] Scanned {lines_scanned:,} variants, found {len(found_rsids)}/{len(target_rsids)} ...")

            cols = line.rstrip("\n").split("\t")
            vcf_id = cols[2]
            rsid = None

            # 1. Try matching by VCF ID directly
            if vcf_id in target_rsids:
                rsid = vcf_id
            else:
                # 2. Try coordinate-based fallback matching
                chrom = cols[0].strip().lower().replace("chr", "")
                try:
                    pos = int(cols[1])
                except ValueError:
                    continue

                if (chrom, pos) in grch37_map:
                    rsid = grch37_map[(chrom, pos)]
                elif (chrom, pos) in grch38_map:
                    rsid = grch38_map[(chrom, pos)]

            if rsid is None:
                continue

            # Prevent duplicate lines for the same rsID in output
            if rsid in found_rsids:
                continue

            # Rewrite VCF ID field to the resolved rsID
            cols[2] = rsid
            found_rsids.add(rsid)

            # Write: first 9 fixed columns + target sample column
            out_cols = cols[:9] + [cols[sample_idx]]
            fout.write("\t".join(out_cols) + "\n")
            print(f"[OK] Found {rsid} via {'coordinate' if vcf_id != rsid else 'rsID'} matching ({len(found_rsids)}/{len(target_rsids)})")

            if len(found_rsids) == len(target_rsids):
                print("[INFO] All HIrisPlex SNPs found — stopping early.")
                break

    print(f"\n[DONE] Wrote {len(found_rsids)} variants to {args.output}")
    print(f"       Scanned {lines_scanned:,} total variant lines.")


if __name__ == "__main__":
    main()

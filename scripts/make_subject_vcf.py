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
    print(f"[INFO] Looking for {len(target_rsids)} HIrisPlex rsIDs")
    print(f"[INFO] Reading {args.merged_vcf} ...")

    sample_idx = None
    found = 0
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
                print(f"[INFO] Scanned {lines_scanned:,} variants, found {found}/{len(target_rsids)} ...")

            # Quick check: does this line contain an rs ID we want?
            cols = line.rstrip("\n").split("\t")
            rsid = cols[2]
            if rsid not in target_rsids:
                continue

            # Write: first 9 fixed columns + target sample column
            out_cols = cols[:9] + [cols[sample_idx]]
            fout.write("\t".join(out_cols) + "\n")
            found += 1
            print(f"[OK] Found {rsid} ({found}/{len(target_rsids)})")

            if found == len(target_rsids):
                print("[INFO] All HIrisPlex SNPs found — stopping early.")
                break

    print(f"\n[DONE] Wrote {found} variants to {args.output}")
    print(f"       Scanned {lines_scanned:,} total variant lines.")


if __name__ == "__main__":
    main()

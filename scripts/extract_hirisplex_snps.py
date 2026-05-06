#!/usr/bin/env python3
"""
extract_hirisplex_snps.py
=========================
Extract HIrisPlex-S 41-SNP dosages from a subject VCF file.

Workflow
--------
1. Load the reference SNP list (hirisplex_41snps.csv) → rsIDs + effect alleles.
2. Parse the subject VCF, index variants by rsID.
3. For each HIrisPlex SNP:
   a. If the rsID is missing from the VCF → record NA, log affected trait model.
   b. If present, compute effect-allele dosage (0 / 1 / 2).
   c. Detect strand flips (complement alleles) and flip dosage accordingly.
4. Write hirisplex_dosages.csv in template column order.
5. Print a coverage report; raise ValueError if < 30 SNPs are present.

Usage
-----
    python scripts/extract_hirisplex_snps.py <subject.vcf>
"""

import argparse
import csv
import gzip
import logging
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_CSV = PROJECT_ROOT / "data" / "reference" / "hirisplex_41snps.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_CSV = OUTPUT_DIR / "hirisplex_dosages.csv"

MIN_SNPS_REQUIRED = 30

# DNA complement map for strand-flip detection
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("extract_hirisplex")


# ---------------------------------------------------------------------------
# Helper: complement an allele string
# ---------------------------------------------------------------------------

def complement_allele(allele: str) -> str:
    """Return the Watson-Crick complement of a single-base allele."""
    return COMPLEMENT.get(allele.upper(), allele)


# ---------------------------------------------------------------------------
# Step 1 — Load the HIrisPlex-S SNP template
# ---------------------------------------------------------------------------

def load_template(template_path: Path) -> list[dict]:
    """
    Parse hirisplex_41snps.csv and return a list of dicts with keys:
        rsID, effect_allele, gene, chromosome, trait_model
    """
    snps = []
    with open(template_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Strip whitespace / BOM from keys
            cleaned = {k.strip().lstrip("\ufeff"): v.strip() for k, v in row.items()}
            snps.append(cleaned)
    log.info("Loaded %d HIrisPlex-S SNPs from template.", len(snps))
    return snps


# ---------------------------------------------------------------------------
# Step 2 — Parse the subject VCF
# ---------------------------------------------------------------------------

def _open_vcf(vcf_path: str):
    """Open a VCF, handling .gz transparently."""
    if vcf_path.endswith(".gz"):
        return gzip.open(vcf_path, "rt")
    return open(vcf_path, "r")


def parse_vcf(vcf_path: str) -> dict:
    """
    Parse a VCF and return a dict keyed by rsID.

    Each value is a dict:
        ref   – REF allele (str)
        alt   – ALT allele (str, first ALT only)
        gt    – genotype tuple, e.g. (0, 1)
    """
    variants: dict[str, dict] = {}
    with _open_vcf(vcf_path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 10:
                continue

            chrom, pos, rsid, ref, alt = cols[0], cols[1], cols[2], cols[3], cols[4]

            # Skip entries without an rsID
            if rsid == "." or not rsid.startswith("rs"):
                continue

            # Use only the first ALT allele
            alt_alleles = alt.split(",")
            first_alt = alt_alleles[0]

            # Parse genotype from the first sample (column 9)
            fmt = cols[8].split(":")
            sample = cols[9].split(":")
            gt_idx = fmt.index("GT") if "GT" in fmt else 0
            gt_str = sample[gt_idx]

            # Handle phased (|) and unphased (/) separators
            sep = "|" if "|" in gt_str else "/"
            try:
                alleles_idx = tuple(int(a) for a in gt_str.split(sep))
            except ValueError:
                # Missing genotype (e.g. "./.")
                continue

            variants[rsid] = {
                "ref": ref.upper(),
                "alt": first_alt.upper(),
                "gt": alleles_idx,
            }
    log.info("Parsed %d rsID-labelled variants from VCF.", len(variants))
    return variants


# ---------------------------------------------------------------------------
# Step 3 — Compute dosage with strand-flip handling
# ---------------------------------------------------------------------------

def _is_strand_flip(vcf_ref: str, vcf_alt: str, effect_allele: str) -> bool:
    """
    Return True if the VCF alleles are the strand complement of what the
    template expects.  E.g. template effect = A, VCF REF = T → complement
    match on the opposite strand.
    """
    comp_ref = complement_allele(vcf_ref)
    comp_alt = complement_allele(vcf_alt)
    # A flip is detected when neither VCF allele matches the effect allele,
    # but the complement of one of them does.
    direct_match = effect_allele in (vcf_ref, vcf_alt)
    complement_match = effect_allele in (comp_ref, comp_alt)
    return (not direct_match) and complement_match


def compute_dosage(
    vcf_entry: dict,
    effect_allele: str,
    rsid: str,
) -> tuple[int, str, str, bool]:
    """
    Compute the dosage of the *effect allele* (0, 1, or 2).

    Returns
    -------
    dosage        : int
    ref_recorded  : str  (REF allele as used, possibly complemented)
    alt_recorded  : str  (ALT allele as used, possibly complemented)
    flipped       : bool (True if a strand flip was applied)
    """
    ref = vcf_entry["ref"]
    alt = vcf_entry["alt"]
    gt = vcf_entry["gt"]

    flipped = False

    if _is_strand_flip(ref, alt, effect_allele):
        # Flip to the complementary strand
        ref = complement_allele(ref)
        alt = complement_allele(alt)
        flipped = True
        log.warning("Strand flip detected for %s: VCF %s/%s → complemented %s/%s",
                     rsid, vcf_entry["ref"], vcf_entry["alt"], ref, alt)

    # Dosage = number of copies of the effect allele in the genotype
    allele_list = [ref] + [alt]  # index 0 = REF, index 1 = first ALT
    dosage = 0
    for idx in gt:
        if idx < len(allele_list) and allele_list[idx] == effect_allele:
            dosage += 1
        elif idx >= len(allele_list):
            # Multi-allelic index beyond what we handle
            pass

    return dosage, ref, alt, flipped


# ---------------------------------------------------------------------------
# Main extraction pipeline
# ---------------------------------------------------------------------------

def extract(vcf_path: str, template_path: Path = TEMPLATE_CSV,
            output_path: Path = OUTPUT_CSV) -> None:
    """Run the full extraction pipeline."""

    # Step 1 — load template
    snp_list = load_template(template_path)
    rsid_order = [s["rsID"] for s in snp_list]

    # Step 2 — parse VCF
    variants = parse_vcf(vcf_path)

    # Step 3 & 4 — compute dosages
    results = {}  # rsID → dosage (int or "NA")
    present_count = 0
    missing_count = 0
    flip_count = 0
    missing_by_trait: dict[str, list[str]] = {}

    for snp in snp_list:
        rsid = snp["rsID"]
        effect = snp["effect_allele"].upper()
        trait = snp.get("trait_model", "unknown")

        if rsid not in variants:
            results[rsid] = {
                "dosage": "NA",
                "ref": "NA",
                "alt": "NA",
                "flipped": False,
            }
            missing_count += 1
            # Log affected trait model(s)
            for t in trait.split(","):
                t = t.strip()
                missing_by_trait.setdefault(t, []).append(rsid)
            log.warning("MISSING %s — affects trait model(s): %s", rsid, trait)
            continue

        dosage, ref_used, alt_used, flipped = compute_dosage(
            variants[rsid], effect, rsid
        )
        results[rsid] = {
            "dosage": dosage,
            "ref": ref_used,
            "alt": alt_used,
            "flipped": flipped,
        }
        present_count += 1
        if flipped:
            flip_count += 1

    # ----- Step 5 — write output CSV in template column order ---------------
    os.makedirs(output_path.parent, exist_ok=True)

    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["rsID", "dosage", "ref_allele", "alt_allele", "strand_flipped"])
        for rsid in rsid_order:
            r = results[rsid]
            writer.writerow([
                rsid,
                r["dosage"],
                r["ref"],
                r["alt"],
                r["flipped"],
            ])

    log.info("Dosage file written to %s", output_path)

    # ----- Step 6 — coverage report ----------------------------------------
    total = len(snp_list)
    print("\n" + "=" * 60)
    print("  HIrisPlex-S SNP Coverage Report")
    print("=" * 60)
    print(f"  Total SNPs in template : {total}")
    print(f"  Present in VCF         : {present_count}")
    print(f"  Missing                : {missing_count}")
    print(f"  Strand flips corrected : {flip_count}")
    print(f"  Coverage               : {present_count / total * 100:.1f}%")
    print("-" * 60)

    if missing_by_trait:
        print("\n  Missing SNPs by affected trait model:")
        for trait, rsids in sorted(missing_by_trait.items()):
            print(f"    {trait:15s} : {', '.join(rsids)}")
    print("=" * 60 + "\n")

    if present_count < MIN_SNPS_REQUIRED:
        raise ValueError(
            f"Insufficient SNP coverage: only {present_count}/{total} SNPs present "
            f"(minimum required: {MIN_SNPS_REQUIRED}). "
            "The HIrisPlex-S prediction will be unreliable."
        )


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract HIrisPlex-S 41-SNP dosages from a subject VCF."
    )
    parser.add_argument(
        "vcf",
        help="Path to the subject VCF file (.vcf or .vcf.gz).",
    )
    parser.add_argument(
        "--template",
        default=str(TEMPLATE_CSV),
        help="Path to hirisplex_41snps.csv (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_CSV),
        help="Path for the output dosage CSV (default: %(default)s).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.vcf):
        log.error("VCF file not found: %s", args.vcf)
        sys.exit(1)

    extract(
        vcf_path=args.vcf,
        template_path=Path(args.template),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()

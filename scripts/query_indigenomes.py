#!/usr/bin/env python3
"""
Query the IndiGenomes web portal for Indian-population allele frequencies at
SNPs required by our pipeline (HIrisPlex-S, facial morphology GWAS loci).

Since IndiGenomes does not offer a bulk VCF download, this script queries
their per-variant web endpoint and also fetches South Asian frequencies from
the Ensembl REST API (gnomAD / 1000 Genomes) as a reliable fallback.

Outputs
-------
    data/reference/indian_allele_frequencies.tsv

Usage
-----
    python scripts/query_indigenomes.py                          # all target SNPs
    python scripts/query_indigenomes.py --snp-list my_snps.txt   # custom list (one rsID per line)
    python scripts/query_indigenomes.py --source ensembl         # skip IndiGenomes, use Ensembl only
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_TSV = PROJECT_ROOT / "data" / "reference" / "indian_allele_frequencies.tsv"
OUTPUT_JSON = PROJECT_ROOT / "data" / "processed" / "allele_freq_query_log.json"

INDIGEN_SEARCH = "https://clingen.igib.res.in/indigen/showdata.php"
ENSEMBL_REST = "https://rest.ensembl.org"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Target SNPs — HIrisPlex-S + key facial morphology GWAS loci
# ---------------------------------------------------------------------------
HIRISPLEX_SNPS = [
    "rs12913832", "rs1800407", "rs12896399", "rs16891982", "rs28777",
    "rs1805008", "rs1805005", "rs1805006", "rs1805007", "rs1805009",
    "rs2228479", "rs1110400", "rs11547464", "rs885479", "rs1426654",
    "rs12203592", "rs1042602", "rs4959270", "rs12821256", "rs2402130",
    "rs12913832", "rs2378249", "rs683",     "rs3114908", "rs1800414",
    "rs10756819", "rs2238289", "rs17128291","rs6497292", "rs1129038",
    "rs1667394",  "rs1126809", "rs1470608", "rs1426654", "rs6119471",
    "rs1545397",  "rs6059655", "rs12441727","rs3212355", "rs8051733",
    "rs2240203",
]

FACIAL_MORPHOLOGY_SNPS = [
    "rs7559271",  "rs1748802",  "rs4648379",  "rs10843104",
    "rs2045323",  "rs17447439", "rs2206437",  "rs6555969",
    "rs11191909", "rs7820428",  "rs4648328",  "rs805722",
    "rs927833",   "rs3827760",  "rs17640804", "rs2894207",
    "rs7567283",  "rs1562005",  "rs12644248", "rs17640804",
]

def get_default_snps() -> list[str]:
    """Deduplicated union of HIrisPlex-S and facial morphology SNPs."""
    return sorted(set(HIRISPLEX_SNPS + FACIAL_MORPHOLOGY_SNPS))


# ===================================================================
# Ensembl REST API — reliable source for SAS allele frequencies
# ===================================================================

def query_ensembl_batch(rs_ids: list[str], batch_size: int = 50) -> dict[str, dict]:
    """
    POST batches of rsIDs to Ensembl and return per-SNP population frequency
    data.  Ensembl enforces rate limits (~15 req/s), so we throttle.
    """
    session = requests.Session()
    session.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
    })

    results: dict[str, dict] = {}
    batches = [rs_ids[i:i + batch_size] for i in range(0, len(rs_ids), batch_size)]

    for batch in tqdm(batches, desc="Ensembl batches", ncols=80):
        url = f"{ENSEMBL_REST}/variation/homo_sapiens"
        payload = json.dumps({
            "ids": batch,
            "population_genotypes": 0,
            "pops": 1,
        })

        for attempt in range(3):
            try:
                resp = session.post(url, data=payload, timeout=60)
                if resp.status_code == 429:
                    wait = float(resp.headers.get("Retry-After", 2))
                    log.warning("Rate limited, waiting %.1fs …", wait)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                log.warning("Attempt %d failed: %s", attempt + 1, exc)
                time.sleep(2 ** attempt)
        else:
            log.error("Giving up on batch starting with %s", batch[0])
            continue

        data = resp.json()
        for rs_id, info in data.items():
            record = _parse_ensembl_variant(rs_id, info)
            if record:
                results[rs_id] = record

        time.sleep(0.5)

    return results


def _parse_ensembl_variant(rs_id: str, info: dict) -> dict | None:
    """Extract chromosome, position, alleles, and population AFs from Ensembl."""
    if not isinstance(info, dict):
        return None

    mappings = info.get("mappings", [])
    if not mappings:
        return None

    m = mappings[0]
    record = {
        "rsid": rs_id,
        "chrom": str(m.get("seq_region_name", "")),
        "pos_grch38": m.get("start"),
        "ref": m.get("allele_string", "").split("/")[0] if "/" in m.get("allele_string", "") else "",
        "alt": m.get("allele_string", "").split("/")[1] if "/" in m.get("allele_string", "") else "",
        "af_global": None,
        "af_sas": None,
        "af_sas_1kg": None,
        "af_gnomad_sas": None,
        "source": "ensembl",
    }

    populations = info.get("populations", [])
    for pop in populations:
        pop_name = pop.get("population", "")
        freq = pop.get("frequency")
        if freq is None:
            continue

        if pop_name == "1000GENOMES:phase_3:SAS" and pop.get("allele", "") != record["ref"]:
            record["af_sas_1kg"] = freq
        if pop_name == "1000GENOMES:phase_3:ALL" and pop.get("allele", "") != record["ref"]:
            record["af_global"] = freq
        if "gnomAD_SAS" in pop_name and pop.get("allele", "") != record["ref"]:
            record["af_gnomad_sas"] = freq

    record["af_sas"] = record["af_sas_1kg"] or record["af_gnomad_sas"]
    return record


# ===================================================================
# IndiGenomes scrape attempt
# ===================================================================

def query_indigenomes(rs_id: str, chrom: str, pos: int, ref: str, alt: str) -> float | None:
    """
    Attempt to pull the IndiGen allele frequency from the portal.
    Returns AF as a float or None if the query fails.
    """
    query_str = f"chr{chrom}:{pos}:{ref}:{alt}"
    try:
        resp = requests.get(
            INDIGEN_SEARCH,
            params={"chr_Start_Ref_Alt": query_str},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        af_match = re.search(r"Allele Frequency.*?([\d.]+)", resp.text)
        if af_match:
            return float(af_match.group(1))
    except (requests.RequestException, ValueError):
        pass
    return None


def enrich_with_indigenomes(records: dict[str, dict]) -> dict[str, dict]:
    """Try to add IndiGenomes AF for each SNP where we have coordinates."""
    log.info("Querying IndiGenomes for %d SNPs (may be slow) …", len(records))
    success = 0
    for rs_id, rec in tqdm(records.items(), desc="IndiGenomes", ncols=80):
        chrom = rec.get("chrom", "")
        pos = rec.get("pos_grch38")
        ref = rec.get("ref", "")
        alt = rec.get("alt", "")
        if not (chrom and pos and ref and alt):
            continue

        af = query_indigenomes(rs_id, chrom, pos, ref, alt)
        if af is not None:
            rec["af_indigen"] = af
            success += 1
        else:
            rec["af_indigen"] = None

        time.sleep(1.0)

    log.info("IndiGenomes: retrieved AF for %d / %d SNPs.", success, len(records))
    return records


# ===================================================================
# Output
# ===================================================================

def save_results(records: dict[str, dict]) -> None:
    """Write allele frequency table as TSV and a JSON log."""
    OUTPUT_TSV.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    rows = list(records.values())
    df = pd.DataFrame(rows)

    col_order = [
        "rsid", "chrom", "pos_grch38", "ref", "alt",
        "af_global", "af_sas", "af_sas_1kg", "af_gnomad_sas",
    ]
    if "af_indigen" in df.columns:
        col_order.append("af_indigen")
    col_order.append("source")
    col_order = [c for c in col_order if c in df.columns]

    df = df[col_order].sort_values(["chrom", "pos_grch38"])
    df.to_csv(OUTPUT_TSV, sep="\t", index=False)
    log.info("Allele frequencies saved → %s  (%d SNPs)", OUTPUT_TSV, len(df))

    meta = {
        "total_snps_queried": len(records),
        "snps_with_sas_af": int(df["af_sas"].notna().sum()) if "af_sas" in df.columns else 0,
        "output_file": str(OUTPUT_TSV),
    }
    with open(OUTPUT_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)


# ===================================================================
# CLI
# ===================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Query allele frequencies for pipeline target SNPs.",
    )
    p.add_argument(
        "--snp-list",
        type=Path,
        default=None,
        help="File with one rsID per line (overrides built-in list).",
    )
    p.add_argument(
        "--source",
        choices=["all", "ensembl"],
        default="all",
        help=(
            "'all' = Ensembl + IndiGenomes scrape; "
            "'ensembl' = Ensembl only (faster, more reliable)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.snp_list:
        snps = [
            line.strip()
            for line in args.snp_list.read_text().splitlines()
            if line.strip().startswith("rs")
        ]
        log.info("Loaded %d SNPs from %s", len(snps), args.snp_list)
    else:
        snps = get_default_snps()
        log.info("Using built-in list: %d target SNPs (HIrisPlex-S + morphology)", len(snps))

    log.info("Querying Ensembl REST API …")
    records = query_ensembl_batch(snps)
    log.info("Ensembl returned data for %d / %d SNPs.", len(records), len(snps))

    if args.source == "all":
        records = enrich_with_indigenomes(records)

    save_results(records)
    log.info("Done.")


if __name__ == "__main__":
    main()

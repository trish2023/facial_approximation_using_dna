#!/usr/bin/env python3
"""
Hybrid ancestry-weighted PRS for the DNA facial reconstruction pipeline.

This module currently operates with Xiong 2025 clumped GWAS traits and remains
fully functional when only European-ancestry effect sizes are available. If
future ancestry-specific Du 2025 summary statistics are added under
``data/processed/gwas_du/``, the same API will automatically blend betas using
the Stage 2 ANI proportion.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import pandas as pd


LOG = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


COMPLEMENT = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
}


@dataclass(frozen=True)
class SNPRecord:
    snp: str
    chrom: str
    pos: int
    effect_allele: str
    other_allele: str
    beta: float


def get_complement(allele: str) -> str:
    return "".join(COMPLEMENT.get(base, base) for base in allele.upper())


def normalize_chrom(chrom: str | int) -> str:
    return str(chrom).strip().lower().removeprefix("chr")


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(result):
        return default
    return result


class AncestryWeightedPRS:
    """Compute ancestry-aware PRS for all available clumped facial traits."""

    def __init__(
        self,
        project_root: Path | None = None,
        gwas_clumped_dir: str = "data/processed/gwas_clumped",
        gwas_cleaned_dir: str = "data/processed/gwas_cleaned",
        ancestry_json_path: str = "data/processed/ancestry_stage2.json",
        ancestry_fallback_path: str = "outputs/ancestry_stage2.json",
        gwas_du_dir: str = "data/processed/gwas_du",
        filtered_vcf_dir: str = "data/processed/1kg_filtered",
        reference_panel_path: str = "data/reference/1000g_panel.txt",
        output_dir: str = "data/processed",
    ) -> None:
        self.project_root = (project_root or PROJECT_ROOT).resolve()
        self.gwas_clumped_dir = self.project_root / gwas_clumped_dir
        self.gwas_cleaned_dir = self.project_root / gwas_cleaned_dir
        self.gwas_du_dir = self.project_root / gwas_du_dir
        self.filtered_vcf_dir = self.project_root / filtered_vcf_dir
        self.reference_panel_path = self.project_root / reference_panel_path
        self.output_dir = self.project_root / output_dir
        self.ancestry_json_path = self.project_root / ancestry_json_path
        self.ancestry_fallback_path = self.project_root / ancestry_fallback_path

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ancestry_info = self._load_ancestry_info()
        self.subject_id = str(self.ancestry_info.get("subject_id", "subject"))
        self.alpha_default = self._extract_alpha(self.ancestry_info)
        self.sas_samples = self._load_sas_samples()
        self.xiong_records = self._load_xiong_records()
        self.du_records = self._load_du_records()
        self.current_trait: str | None = None

        LOG.info(
            "Initialised PRS module for subject %s with %d clumped traits.",
            self.subject_id,
            len(self.xiong_records),
        )

    def blend_betas(self, alpha: float, trait: str | None = None) -> dict[str, float]:
        """
        Blend Xiong and future Du betas for one trait.

        Returns a mapping of SNP -> blended beta and records SNP overlap
        metadata on ``self._last_blend_stats`` for downstream reporting.
        """
        trait_name = self._resolve_trait_context(trait)
        eur = self.xiong_records.get(trait_name, {})
        if not eur:
            raise KeyError(f"No Xiong lead-SNP betas loaded for trait '{trait_name}'.")

        du = self.du_records.get(trait_name, {})
        blended: dict[str, float] = {}
        shared_snps = 0
        eur_only_snps = 0
        du_only_snps = 0

        eur_keys = set(eur)
        du_keys = set(du)

        for snp in sorted(eur_keys | du_keys):
            eur_record = eur.get(snp)
            du_record = du.get(snp)

            if eur_record and du_record:
                beta_star = alpha * eur_record.beta + (1.0 - alpha) * du_record.beta
                shared_snps += 1
            elif eur_record:
                beta_star = eur_record.beta
                eur_only_snps += 1
            else:
                beta_star = du_record.beta
                du_only_snps += 1

            blended[snp] = beta_star

        self._last_blend_stats = {
            "trait": trait_name,
            "shared_snps": shared_snps,
            "eur_only_snps": eur_only_snps,
            "du_only_snps": du_only_snps,
            "du_beta_available": bool(du),
        }
        return blended

    def compute_prs(
        self,
        subject_dosages: dict[str, float | None],
        alpha: float,
        trait: str | None = None,
        imputation_values: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        Compute PRS for a single subject/sample against one trait.

        ``subject_dosages`` maps SNP IDs to dosage values. Missing values may be
        ``None``; these are mean-imputed using ``imputation_values`` when
        available, otherwise zero-imputed.
        """
        beta_map = self.blend_betas(alpha, trait)
        imputation_values = imputation_values or {}

        prs_value = 0.0
        snp_count = 0
        imputed_count = 0

        for snp, beta_star in beta_map.items():
            dosage = subject_dosages.get(snp)
            if dosage is None or (isinstance(dosage, float) and math.isnan(dosage)):
                dosage = imputation_values.get(snp, 0.0)
                imputed_count += 1
            prs_value += safe_float(dosage) * beta_star
            snp_count += 1

        blend_stats = getattr(self, "_last_blend_stats", {})
        return {
            "prs": prs_value,
            "snp_count": snp_count,
            "imputed_count": imputed_count,
            "shared_snps": int(blend_stats.get("shared_snps", 0)),
            "eur_only": int(blend_stats.get("eur_only_snps", 0)),
            "du_only": int(blend_stats.get("du_only_snps", 0)),
            "alpha_used": alpha,
        }

    def run_all_traits(self) -> dict[str, dict[str, float | str]]:
        """Run PRS scoring, SAS normalisation, and output generation."""
        results: dict[str, dict[str, float | str]] = {}
        raw_scores: dict[str, dict[str, float]] = {}
        reference_stats: dict[str, dict[str, float | int]] = {}
        zscore_payload: dict[str, dict[str, float | str]] = {}
        summary_rows: list[dict[str, float | str]] = []

        trait_positions = self._collect_trait_positions()
        genotype_cache = self._load_genotypes_for_positions(trait_positions)

        for trait in sorted(self.xiong_records):
            self.current_trait = trait
            alpha = self.alpha_default
            trait_snp_ids = set(self.xiong_records[trait]) | set(self.du_records.get(trait, {}))

            subject_dosages, sas_dosages = self._build_trait_dosage_maps(
                trait=trait,
                genotype_cache=genotype_cache,
                trait_snp_ids=trait_snp_ids,
            )
            imputation_values = self._compute_imputation_values(sas_dosages)

            trait_result = self.compute_prs(
                subject_dosages=subject_dosages,
                alpha=alpha,
                trait=trait,
                imputation_values=imputation_values,
            )
            raw_scores[trait] = {"prs": trait_result["prs"], "alpha_used": alpha}

            sas_prs_values = [
                self.compute_prs(
                    subject_dosages=sample_dosages,
                    alpha=alpha,
                    imputation_values=imputation_values,
                )["prs"]
                for sample_dosages in sas_dosages.values()
            ]

            mean_prs = mean(sas_prs_values) if sas_prs_values else 0.0
            sd_prs = pstdev(sas_prs_values) if len(sas_prs_values) > 1 else 0.0
            z_score = (trait_result["prs"] - mean_prs) / sd_prs if sd_prs > 0 else 0.0
            interpretation = self._interpret_zscore(z_score)

            reference_stats[trait] = {
                "mean_prs": mean_prs,
                "sd_prs": sd_prs,
                "reference_n": len(sas_prs_values),
            }
            zscore_payload[trait] = {
                "prs": trait_result["prs"],
                "z_score": z_score,
                "interpretation": interpretation,
                "alpha_used": alpha,
            }

            result_row = {
                **trait_result,
                "trait": trait,
                "mean_prs": mean_prs,
                "sd_prs": sd_prs,
                "z_score": z_score,
                "interpretation": interpretation,
            }
            results[trait] = result_row
            summary_rows.append(
                {
                    "trait": trait,
                    "prs": trait_result["prs"],
                    "z_score": z_score,
                    "snp_count": trait_result["snp_count"],
                    "imputed_count": trait_result["imputed_count"],
                    "shared_snps": trait_result["shared_snps"],
                    "eur_only": trait_result["eur_only"],
                    "du_only": trait_result["du_only"],
                    "alpha_used": trait_result["alpha_used"],
                }
            )

        self._write_json(self.output_dir / "prs_scores.json", results)
        self._write_json(self.output_dir / "prs_reference_stats.json", reference_stats)
        self._write_json(self.output_dir / "prs_zscores.json", zscore_payload)
        self._write_summary_csv(self.output_dir / "prs_summary.csv", summary_rows)

        LOG.info("Saved PRS outputs under %s", self.output_dir)
        return results

    def _resolve_trait_context(self, trait: str | None) -> str:
        if trait is not None:
            return trait
        if self.current_trait is not None:
            return self.current_trait
        raise ValueError("Trait context is not set. Pass trait=... or set self.current_trait first.")

    def _load_ancestry_info(self) -> dict[str, object]:
        ancestry_path = self.ancestry_json_path
        if not ancestry_path.exists():
            ancestry_path = self.ancestry_fallback_path
        if not ancestry_path.exists():
            raise FileNotFoundError(
                "Stage 2 ancestry JSON not found at "
                f"{self.ancestry_json_path} or {self.ancestry_fallback_path}."
            )
        with ancestry_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        LOG.info("Loaded ancestry proportions from %s", ancestry_path)
        return payload

    def _extract_alpha(self, ancestry_info: dict[str, object]) -> float:
        alpha = ancestry_info.get("ani_proportion")
        if alpha is None:
            probs = ancestry_info.get("probabilities", {})
            if isinstance(probs, dict):
                for key, value in probs.items():
                    if "indoaryan" in key.lower():
                        alpha = value
                        break
        alpha = safe_float(alpha, 1.0)
        return min(max(alpha, 0.0), 1.0)

    def _load_sas_samples(self) -> list[str]:
        panel = pd.read_csv(self.reference_panel_path, sep="\t")
        panel.columns = [str(col).strip() for col in panel.columns]
        if "sample" not in panel.columns or "super_pop" not in panel.columns:
            raise ValueError("1000 Genomes panel file must contain sample and super_pop columns.")
        sas_samples = panel.loc[panel["super_pop"] == "SAS", "sample"].astype(str).tolist()
        if not sas_samples:
            raise ValueError("No SAS samples found in the reference panel.")
        return sas_samples

    def _load_xiong_records(self) -> dict[str, dict[str, SNPRecord]]:
        lead_files = sorted(self.gwas_clumped_dir.glob("*_lead_snps.txt"))
        if not lead_files:
            raise FileNotFoundError(f"No clumped lead SNP files found in {self.gwas_clumped_dir}.")

        trait_records: dict[str, dict[str, SNPRecord]] = {}
        for lead_file in lead_files:
            trait = lead_file.stem.replace("_lead_snps", "")
            cleaned_path = self.gwas_cleaned_dir / f"{trait}_cleaned.txt"
            if not cleaned_path.exists():
                raise FileNotFoundError(f"Missing cleaned GWAS file for trait {trait}: {cleaned_path}")

            lead_snps = {
                line.strip()
                for line in lead_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            df = pd.read_csv(cleaned_path, sep="\t")
            missing_columns = {"SNP", "CHR", "BP", "A1", "A2", "BETA"} - set(df.columns)
            if missing_columns:
                raise ValueError(f"{cleaned_path} is missing columns: {sorted(missing_columns)}")

            records: dict[str, SNPRecord] = {}
            for _, row in df[df["SNP"].isin(lead_snps)].iterrows():
                snp_id = str(row["SNP"])
                records[snp_id] = SNPRecord(
                    snp=snp_id,
                    chrom=normalize_chrom(row["CHR"]),
                    pos=int(row["BP"]),
                    effect_allele=str(row["A1"]).upper(),
                    other_allele=str(row["A2"]).upper(),
                    beta=float(row["BETA"]),
                )
            if not records:
                raise ValueError(f"No lead SNP betas recovered for trait {trait}.")
            trait_records[trait] = records
        return trait_records

    def _load_du_records(self) -> dict[str, dict[str, SNPRecord]]:
        if not self.gwas_du_dir.exists():
            LOG.info("Du beta directory not present at %s; using Xiong-only PRS.", self.gwas_du_dir)
            return {}

        trait_records: dict[str, dict[str, SNPRecord]] = {}
        candidate_files = sorted(
            path
            for path in self.gwas_du_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".txt", ".tsv", ".csv"}
        )
        for path in candidate_files:
            try:
                trait_name, records = self._parse_du_trait_file(path)
            except Exception as exc:
                LOG.warning("Skipping Du file %s: %s", path.name, exc)
                continue
            if trait_name in self.xiong_records and records:
                trait_records[trait_name] = records

        if trait_records:
            LOG.info("Loaded ancestry-specific Du betas for %d traits.", len(trait_records))
        else:
            LOG.info("No usable Du beta files detected in %s.", self.gwas_du_dir)
        return trait_records

    def _parse_du_trait_file(self, path: Path) -> tuple[str, dict[str, SNPRecord]]:
        sep = "\t" if path.suffix.lower() in {".txt", ".tsv"} else ","
        df = pd.read_csv(path, sep=sep)
        normalised = {self._canonical_column(col): col for col in df.columns}

        required_aliases = {
            "snp": ["snp", "variant_id", "id", "markername", "rsid"],
            "chr": ["chr", "chrom", "chromosome"],
            "bp": ["bp", "pos", "position"],
            "a1": ["a1", "ea", "effect_allele", "alt"],
            "a2": ["a2", "nea", "other_allele", "ref"],
            "beta": ["beta", "effect", "estimate"],
        }
        resolved = {}
        for canonical, aliases in required_aliases.items():
            match = next((normalised[a] for a in aliases if a in normalised), None)
            if match is None:
                raise ValueError(f"Missing required column for {canonical}")
            resolved[canonical] = match

        trait_name = self._resolve_du_trait_name(path, df)
        target_snps = set(self.xiong_records.get(trait_name, {}))
        if not target_snps:
            raise ValueError(f"Trait {trait_name} does not match any clumped Xiong trait.")

        if "snp" in resolved:
            subset = df[df[resolved["snp"]].astype(str).isin(target_snps)].copy()
        else:
            subset = df.copy()

        records: dict[str, SNPRecord] = {}
        for _, row in subset.iterrows():
            snp = str(row[resolved["snp"]]).strip()
            if snp not in target_snps:
                continue
            records[snp] = SNPRecord(
                snp=snp,
                chrom=normalize_chrom(row[resolved["chr"]]),
                pos=int(row[resolved["bp"]]),
                effect_allele=str(row[resolved["a1"]]).upper(),
                other_allele=str(row[resolved["a2"]]).upper(),
                beta=float(row[resolved["beta"]]),
            )
        return trait_name, records

    def _resolve_du_trait_name(self, path: Path, df: pd.DataFrame) -> str:
        for column in df.columns:
            canonical = self._canonical_column(column)
            if canonical == "trait":
                trait_value = str(df.iloc[0][column]).strip()
                mapped = self._match_trait_name(trait_value)
                if mapped:
                    return mapped
        mapped = self._match_trait_name(path.stem)
        if mapped:
            return mapped
        raise ValueError(f"Could not map {path.name} to a known clumped Xiong trait.")

    def _match_trait_name(self, raw_name: str) -> str | None:
        raw_key = self._trait_key(raw_name)
        for trait_name in self.xiong_records:
            if self._trait_key(trait_name) == raw_key:
                return trait_name
        return None

    def _trait_key(self, value: str) -> str:
        simplified = re.sub(r"^(xiong2025_|du2025_)", "", value.strip().lower())
        return re.sub(r"[^a-z0-9]+", "", simplified)

    def _canonical_column(self, column: object) -> str:
        return re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")

    def _collect_trait_positions(self) -> dict[str, dict[int, list[SNPRecord]]]:
        positions: dict[str, dict[int, list[SNPRecord]]] = {}
        for trait_map in [self.xiong_records, self.du_records]:
            for records in trait_map.values():
                for record in records.values():
                    chrom_bucket = positions.setdefault(record.chrom, {})
                    chrom_bucket.setdefault(record.pos, []).append(record)
        return positions

    def _load_genotypes_for_positions(
        self,
        positions_by_chrom: dict[str, dict[int, list[SNPRecord]]],
    ) -> dict[str, dict[str, dict[str, float | None]]]:
        genotype_cache: dict[str, dict[str, dict[str, float | None]]] = {}
        for chrom, positions in positions_by_chrom.items():
            vcf_path = self.filtered_vcf_dir / f"chr{chrom}_SAS_EUR_filtered.vcf.gz"
            if not vcf_path.exists():
                raise FileNotFoundError(f"Required filtered VCF not found: {vcf_path}")
            self._scan_vcf_for_positions(vcf_path, chrom, positions, genotype_cache)
        return genotype_cache

    def _scan_vcf_for_positions(
        self,
        vcf_path: Path,
        chrom: str,
        target_positions: dict[int, list[SNPRecord]],
        genotype_cache: dict[str, dict[str, dict[str, float | None]]],
    ) -> None:
        sorted_positions = sorted(target_positions)
        max_target = sorted_positions[-1]
        remaining = set(sorted_positions)
        sample_names: list[str] | None = None
        subject_idx: int | None = None
        sas_indices: list[tuple[str, int]] = []

        with gzip.open(vcf_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    header = line.rstrip("\n").split("\t")
                    sample_names = header[9:]
                    try:
                        subject_idx = sample_names.index(self.subject_id)
                    except ValueError as exc:
                        raise ValueError(
                            f"Subject {self.subject_id} is not present in {vcf_path.name}."
                        ) from exc
                    sample_to_idx = {sample: idx for idx, sample in enumerate(sample_names)}
                    sas_indices = [
                        (sample, sample_to_idx[sample])
                        for sample in self.sas_samples
                        if sample in sample_to_idx
                    ]
                    continue

                if sample_names is None or subject_idx is None:
                    raise ValueError(f"Malformed VCF header in {vcf_path}")

                fields = line.rstrip("\n").split("\t")
                pos = int(fields[1])
                if pos > max_target:
                    break
                if pos not in remaining:
                    continue

                ref = fields[3].upper()
                alt = fields[4].split(",")[0].upper()
                if "," in fields[4]:
                    LOG.debug("Skipping multi-allelic variant at %s:%s", chrom, pos)
                    remaining.remove(pos)
                    continue

                format_keys = fields[8].split(":")
                gt_index = format_keys.index("GT") if "GT" in format_keys else 0
                sample_fields = fields[9:]

                for record in target_positions[pos]:
                    subject_gt = sample_fields[subject_idx]
                    subject_dosage = self._extract_effect_dosage(
                        genotype_field=subject_gt,
                        gt_index=gt_index,
                        ref=ref,
                        alt=alt,
                        record=record,
                    )

                    sample_dosages = {
                        sample: self._extract_effect_dosage(
                            genotype_field=sample_fields[sample_idx],
                            gt_index=gt_index,
                            ref=ref,
                            alt=alt,
                            record=record,
                        )
                        for sample, sample_idx in sas_indices
                    }

                    genotype_cache[record.snp] = {
                        "subject": {self.subject_id: subject_dosage},
                        "sas": sample_dosages,
                    }
                remaining.remove(pos)
                if not remaining:
                    break

        for pos in sorted(remaining):
            for record in target_positions[pos]:
                genotype_cache.setdefault(
                    record.snp,
                    {
                        "subject": {self.subject_id: None},
                        "sas": {sample: None for sample in self.sas_samples},
                    },
                )
                LOG.warning("Variant %s (%s:%d) not found in %s", record.snp, chrom, pos, vcf_path.name)

    def _extract_effect_dosage(
        self,
        genotype_field: str,
        gt_index: int,
        ref: str,
        alt: str,
        record: SNPRecord,
    ) -> float | None:
        if not genotype_field:
            return None
        pieces = genotype_field.split(":")
        gt = pieces[gt_index] if gt_index < len(pieces) else pieces[0]
        if "." in gt:
            return None

        allele_codes = re.split(r"[|/]", gt)
        allele_map = {"0": ref, "1": alt}

        same_strand = {record.effect_allele, record.other_allele} == {ref, alt}
        comp_effect = get_complement(record.effect_allele)
        comp_other = get_complement(record.other_allele)
        comp_strand = {comp_effect, comp_other} == {ref, alt}

        if same_strand:
            target_allele = record.effect_allele
        elif comp_strand:
            target_allele = comp_effect
        else:
            return None

        dosage = 0
        for code in allele_codes:
            allele = allele_map.get(code)
            if allele is None:
                return None
            if allele == target_allele:
                dosage += 1
        return float(dosage)

    def _build_trait_dosage_maps(
        self,
        trait: str,
        genotype_cache: dict[str, dict[str, dict[str, float | None]]],
        trait_snp_ids: Iterable[str],
    ) -> tuple[dict[str, float | None], dict[str, dict[str, float | None]]]:
        subject_dosages: dict[str, float | None] = {}
        sas_dosages: dict[str, dict[str, float | None]] = {
            sample: {} for sample in self.sas_samples
        }

        for snp in sorted(set(trait_snp_ids)):
            cached = genotype_cache.get(snp)
            if not cached:
                subject_dosages[snp] = None
                for sample in self.sas_samples:
                    sas_dosages[sample][snp] = None
                continue

            subject_dosages[snp] = cached["subject"].get(self.subject_id)
            for sample in self.sas_samples:
                sas_dosages[sample][snp] = cached["sas"].get(sample)

        return subject_dosages, sas_dosages

    def _compute_imputation_values(
        self,
        sas_dosages: dict[str, dict[str, float | None]],
    ) -> dict[str, float]:
        per_snp_values: dict[str, list[float]] = {}
        for sample_map in sas_dosages.values():
            for snp, dosage in sample_map.items():
                if dosage is not None:
                    per_snp_values.setdefault(snp, []).append(float(dosage))
        return {
            snp: (sum(values) / len(values) if values else 0.0)
            for snp, values in per_snp_values.items()
        }

    def _interpret_zscore(self, z_score: float) -> str:
        abs_z = abs(z_score)
        if abs_z < 1:
            return "Mild"
        if abs_z < 2:
            return "Moderate"
        return "Strong"

    def _write_json(self, path: Path, payload: object) -> None:
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _write_summary_csv(self, path: Path, rows: list[dict[str, float | str]]) -> None:
        fieldnames = [
            "trait",
            "prs",
            "z_score",
            "snp_count",
            "imputed_count",
            "shared_snps",
            "eur_only",
            "du_only",
            "alpha_used",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    prs = AncestryWeightedPRS()
    prs.run_all_traits()


if __name__ == "__main__":
    main()

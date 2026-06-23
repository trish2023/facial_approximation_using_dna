#!/usr/bin/env python3
"""
Master orchestration script for the DNA Facial Approximation pipeline.

This controller wires together the existing scientific scripts, checks that the
workspace is ready, and mirrors the final artifacts into a user-selected output
directory. It does not reimplement any model logic.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"
DEFAULT_VCF = PROJECT_ROOT / "data" / "processed" / "subject_hirisplex.vcf"

REQUIRED_RESOURCES = [
    ("reference SNP file", PROJECT_ROOT / "data" / "reference" / "hirisplex_41snps.csv"),
    ("reference SNP file", PROJECT_ROOT / "data" / "reference" / "hirisplex_coefficients.json"),
    ("reference panel file", PROJECT_ROOT / "data" / "reference" / "1000g_panel.txt"),
    ("face template", PROJECT_ROOT / "data" / "reference" / "face_templates" / "indoaryan_male.jpg"),
    ("face template", PROJECT_ROOT / "data" / "reference" / "face_templates" / "indoaryan_female.jpg"),
    ("face template", PROJECT_ROOT / "data" / "reference" / "face_templates" / "dravidian_male.jpg"),
    ("face template", PROJECT_ROOT / "data" / "reference" / "face_templates" / "dravidian_female.jpg"),
    ("dlib predictor", PROJECT_ROOT / "data" / "reference" / "dlib" / "shape_predictor_68_face_landmarks.dat"),
    ("ancestry model", PROJECT_ROOT / "models" / "stage2_rf.pkl"),
    ("ancestry model", PROJECT_ROOT / "models" / "superpop_classifier.pkl"),
]

REQUIRED_SCRIPT_FILES = [
    PROJECT_ROOT / "scripts" / "superpop_classifier.py",
    PROJECT_ROOT / "scripts" / "stage2_random_forest.py",
    PROJECT_ROOT / "scripts" / "predict_sex.py",
    PROJECT_ROOT / "scripts" / "extract_hirisplex_snps.py",
    PROJECT_ROOT / "scripts" / "run_hirisplex.py",
    PROJECT_ROOT / "scripts" / "pigmentation_uncertainty.py",
    PROJECT_ROOT / "scripts" / "weighted_prs.py",
    PROJECT_ROOT / "scripts" / "normalise_prs.py",
    PROJECT_ROOT / "scripts" / "template_selection.py",
    PROJECT_ROOT / "scripts" / "landmark_detection.py",
    PROJECT_ROOT / "scripts" / "landmark_displacement_mapping.py",
    PROJECT_ROOT / "scripts" / "delaunay_warp.py",
    PROJECT_ROOT / "scripts" / "apply_pigmentation.py",
    PROJECT_ROOT / "scripts" / "generate_qc_report.py",
]

REQUIRED_OUTPUT_DIRECTORIES = [
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "data" / "processed",
]


def script_path(relative: str) -> Path:
    return PROJECT_ROOT / relative


def load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, found {type(payload).__name__}.")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def read_vcf_sample_id(vcf_path: Path) -> str:
    opener = gzip.open if vcf_path.suffix == ".gz" else open
    with opener(vcf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                if len(cols) <= 9:
                    raise ValueError(f"No sample columns found in VCF: {vcf_path}")
                return cols[9]
    raise ValueError(f"VCF header not found in {vcf_path}")


def ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def ensure_preflight(vcf_path: Path, config_path: Path, output_dir: Path) -> None:
    ensure_file(vcf_path, "input VCF")
    ensure_file(config_path, "config file")
    for label, path in REQUIRED_RESOURCES:
        ensure_file(path, label)
    for path in REQUIRED_SCRIPT_FILES:
        ensure_file(path, "pipeline script")
    for path in [*REQUIRED_OUTPUT_DIRECTORIES, output_dir]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_stage_outputs(stage: "PipelineStage") -> None:
    missing_outputs = [str(path) for path in stage.outputs if not path.exists()]
    if missing_outputs:
        joined = "\n  - ".join(missing_outputs)
        raise FileNotFoundError(f"Stage {stage.name} completed but did not produce expected output(s):\n  - {joined}")


def mirror_files(src_paths: list[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src in src_paths:
        if not src.exists():
            continue
        dest = output_dir / src.name
        if src.resolve() == dest.resolve():
            continue
        shutil.copy2(src, dest)


def ansi_safe_json(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


@dataclass(frozen=True)
class PipelineStage:
    name: str
    command: list[str]
    outputs: list[Path]
    optional: bool = False


class PipelineController:
    def __init__(
        self,
        vcf_path: Path,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        config_path: Path = DEFAULT_CONFIG,
        skip_stage2: bool = False,
        dry_run: bool = False,
    ) -> None:
        self.vcf_path = vcf_path.resolve()
        self.output_dir = output_dir.resolve()
        self.config_path = config_path.resolve()
        self.skip_stage2 = skip_stage2
        self.dry_run = dry_run
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "pipeline.log"
        self.failed_path = self.output_dir / "PIPELINE_FAILED.json"
        self.stage_results: list[dict[str, Any]] = []
        self.subject_id = read_vcf_sample_id(self.vcf_path)
        self._logger = self._configure_logging()
        self._planned_stages = self._build_stages()

    def _configure_logging(self) -> logging.Logger:
        logger = logging.getLogger(f"pipeline.{id(self)}")
        logger.setLevel(logging.INFO)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

        fmt = logging.Formatter("%(asctime)s  [%(levelname)s]  %(message)s", "%Y-%m-%d %H:%M:%S")
        file_handler = logging.FileHandler(self.log_path, mode="w", encoding="utf-8")
        file_handler.setFormatter(fmt)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
        return logger

    def _build_stages(self) -> list[PipelineStage]:
        stages: list[PipelineStage] = [
            PipelineStage(
                name="ancestry_superpopulation",
                command=[sys.executable, str(script_path("scripts/superpop_classifier.py")), "--subject", self.subject_id],
                outputs=[PROJECT_ROOT / "outputs" / "ancestry_stage1.json"],
            ),
        ]

        if not self.skip_stage2:
            stages.append(
                PipelineStage(
                    name="ancestry_subclassification",
                    command=[sys.executable, str(script_path("scripts/stage2_random_forest.py"))],
                    outputs=[PROJECT_ROOT / "outputs" / "ancestry_stage2.json"],
                )
            )

        stages.extend(
            [
                PipelineStage(
                    name="sex_prediction",
                    command=[sys.executable, str(script_path("scripts/predict_sex.py")), str(self.vcf_path)],
                    outputs=[PROJECT_ROOT / "outputs" / "sex_prediction.json"],
                ),
                PipelineStage(
                    name="extract_hirisplex_dosages",
                    command=[sys.executable, str(script_path("scripts/extract_hirisplex_snps.py")), str(self.vcf_path)],
                    outputs=[PROJECT_ROOT / "outputs" / "hirisplex_dosages.csv"],
                ),
                PipelineStage(
                    name="run_hirisplex_predictions",
                    command=[sys.executable, str(script_path("scripts/run_hirisplex.py"))],
                    outputs=[PROJECT_ROOT / "outputs" / "hirisplex_predictions.json"],
                ),
                PipelineStage(
                    name="pigmentation_uncertainty",
                    command=[sys.executable, str(script_path("scripts/pigmentation_uncertainty.py"))],
                    outputs=[PROJECT_ROOT / "outputs" / "pigmentation_report.json"],
                ),
                PipelineStage(
                    name="prs_computation",
                    command=[sys.executable, str(script_path("scripts/weighted_prs.py"))],
                    outputs=[
                        PROJECT_ROOT / "data" / "processed" / "prs_scores.json",
                        PROJECT_ROOT / "data" / "processed" / "prs_zscores.json",
                        PROJECT_ROOT / "data" / "processed" / "prs_summary.csv",
                        PROJECT_ROOT / "outputs" / "prs_scores.json",
                        PROJECT_ROOT / "outputs" / "prs_zscores.json",
                        PROJECT_ROOT / "outputs" / "prs_summary.csv",
                    ],
                ),
                PipelineStage(
                    name="morphology_generation",
                    command=[sys.executable, str(script_path("scripts/normalise_prs.py"))],
                    outputs=[
                        PROJECT_ROOT / "data" / "processed" / "morphology_traits.json",
                        PROJECT_ROOT / "data" / "processed" / "top_traits.json",
                        PROJECT_ROOT / "outputs" / "morphology_traits.json",
                        PROJECT_ROOT / "outputs" / "top_traits.json",
                    ],
                ),
                PipelineStage(
                    name="template_selection",
                    command=[sys.executable, str(script_path("scripts/template_selection.py"))],
                    outputs=[PROJECT_ROOT / "outputs" / "template_selection.json"],
                ),
                PipelineStage(
                    name="landmark_detection",
                    command=[sys.executable, str(script_path("scripts/landmark_detection.py"))],
                    outputs=[
                        PROJECT_ROOT / "data" / "processed" / "template_landmarks.json",
                        PROJECT_ROOT / "outputs" / "template_landmarks_overlay.png",
                    ],
                ),
                PipelineStage(
                    name="displacement_mapping",
                    command=[sys.executable, str(script_path("scripts/landmark_displacement_mapping.py"))],
                    outputs=[
                        PROJECT_ROOT / "data" / "processed" / "landmark_displacements.json",
                        PROJECT_ROOT / "outputs" / "displacement_summary.json",
                    ],
                ),
                PipelineStage(
                    name="delaunay_warp",
                    command=[sys.executable, str(script_path("scripts/delaunay_warp.py"))],
                    outputs=[
                        PROJECT_ROOT / "outputs" / "morphology_warped_face.png",
                        PROJECT_ROOT / "outputs" / "warp_qc.json",
                    ],
                ),
                PipelineStage(
                    name="apply_pigmentation",
                    command=[sys.executable, str(script_path("scripts/apply_pigmentation.py"))],
                    outputs=[
                        PROJECT_ROOT / "outputs" / "pigmentation_applied_face.png",
                        PROJECT_ROOT / "outputs" / "pigmentation_qc.json",
                    ],
                ),
                PipelineStage(
                    name="qc_report",
                    command=[sys.executable, str(script_path("scripts/generate_qc_report.py"))],
                    outputs=[
                        PROJECT_ROOT / "outputs" / "qc_report.json",
                        PROJECT_ROOT / "outputs" / "qc_report.md",
                    ],
                ),
            ]
        )
        return stages

    def _write_stage2_compat_from_stage1(self) -> Path:
        stage1_path = PROJECT_ROOT / "outputs" / "ancestry_stage1.json"
        stage2_path = PROJECT_ROOT / "outputs" / "ancestry_stage2.json"
        payload = load_json_object(stage1_path)
        probs = payload.get("probabilities", {})
        sas_prob = float(probs.get("SAS", 0.0) or 0.0)
        eur_prob = float(probs.get("EUR", 0.0) or 0.0)
        predicted_superpop = str(payload.get("predicted_superpopulation", "SAS"))
        predicted_label = "IndoAryan-proxy" if predicted_superpop == "SAS" else "Dravidian-proxy"
        compat = {
            "subject_id": payload.get("subject_id"),
            "predicted_label": predicted_label,
            "probabilities": {
                "IndoAryan-proxy": sas_prob,
                "Dravidian-proxy": eur_prob,
            },
            "ani_proportion": sas_prob,
            "asi_proportion": eur_prob,
            "source": "stage1_compatibility",
            "skip_stage2": True,
        }
        write_json(stage2_path, compat)
        return stage2_path

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )

    def _record_failure(self, stage_name: str, error: str) -> None:
        payload = {
            "failed_stage": stage_name,
            "error": error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        write_json(self.failed_path, payload)

    def _log_stage_output(self, stdout: str, stderr: str) -> None:
        if stdout.strip():
            for line in stdout.rstrip().splitlines():
                self._logger.info(line)
        if stderr.strip():
            for line in stderr.rstrip().splitlines():
                self._logger.error(line)

    def _run_stage(self, stage: PipelineStage) -> None:
        start = time.perf_counter()
        self._logger.info("Starting stage: %s", stage.name)
        self._logger.info("Command: %s", " ".join(stage.command))
        result = self._run_command(stage.command)
        self._log_stage_output(result.stdout, result.stderr)
        if result.returncode != 0:
            raise RuntimeError(
                f"Stage {stage.name} failed with exit code {result.returncode}. "
                f"{result.stderr.strip() or result.stdout.strip() or 'No additional output.'}"
            )
        elapsed = time.perf_counter() - start
        self._logger.info("Completed stage: %s in %.2f seconds", stage.name, elapsed)
        if stage.name == "prs_computation":
            mirror_files(
                [
                    PROJECT_ROOT / "data" / "processed" / "prs_scores.json",
                    PROJECT_ROOT / "data" / "processed" / "prs_zscores.json",
                    PROJECT_ROOT / "data" / "processed" / "prs_summary.csv",
                ],
                PROJECT_ROOT / "outputs",
            )
        elif stage.name == "morphology_generation":
            mirror_files(
                [
                    PROJECT_ROOT / "data" / "processed" / "morphology_traits.json",
                    PROJECT_ROOT / "data" / "processed" / "top_traits.json",
                ],
                PROJECT_ROOT / "outputs",
            )
        ensure_stage_outputs(stage)
        mirror_files(stage.outputs, self.output_dir)
        if stage.name == "ancestry_superpopulation" and self.skip_stage2:
            compat_path = self._write_stage2_compat_from_stage1()
            mirror_files([compat_path], self.output_dir)

    def _run_optional_dry_run_stage(self, stage: PipelineStage) -> None:
        self._logger.info("Planned stage: %s", stage.name)
        self._logger.info("Command: %s", " ".join(stage.command))

    def run(self) -> dict[str, Any]:
        ensure_preflight(self.vcf_path, self.config_path, self.output_dir)

        if self.dry_run:
            self._logger.info("Dry run enabled; validating execution order only.")
            for stage in self._planned_stages:
                self._run_optional_dry_run_stage(stage)
            return {
                "status": "dry_run",
                "execution_order": [stage.name for stage in self._planned_stages],
                "subject_id": self.subject_id,
            }

        run_started = time.perf_counter()
        completed: list[str] = []
        current_stage = "initialisation"
        try:
            for stage in self._planned_stages:
                current_stage = stage.name
                self._run_stage(stage)
                completed.append(stage.name)
        except Exception as exc:
            self._logger.error("Pipeline failed in stage %s: %s", current_stage, exc)
            self._record_failure(current_stage, str(exc))
            raise

        total_runtime = time.perf_counter() - run_started
        qc_report_path = self.output_dir / "qc_report.json"
        qc_payload = load_json_object(qc_report_path) if qc_report_path.exists() else {}
        overall_reliability = (
            qc_payload.get("overview", {}).get("overall_reliability_score")
            if isinstance(qc_payload.get("overview"), dict)
            else None
        )

        summary = {
            "status": "success",
            "completed_stages": completed,
            "total_runtime_seconds": round(total_runtime, 2),
            "final_output": str(self.output_dir / "pigmentation_applied_face.png"),
            "qc_report": str(qc_report_path),
            "overall_reliability_score": overall_reliability,
        }
        self._logger.info("Pipeline completed successfully")
        self._logger.info("Stages completed: %s", completed)
        self._logger.info("Total runtime: %.2f seconds", total_runtime)
        self._logger.info("Final output: %s", summary["final_output"])
        self._logger.info("QC report: %s", summary["qc_report"])
        self._logger.info("Overall reliability score: %s", overall_reliability)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DNA Facial Approximation pipeline.")
    parser.add_argument("--vcf", required=True, help="Path to the subject VCF input.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for final outputs (default: %(default)s).")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Path to config.yaml (default: %(default)s).")
    parser.add_argument("--skip-stage2", action="store_true", help="Skip South Asian sub-classification.")
    parser.add_argument("--dry-run", action="store_true", help="Validate dependencies and execution order only.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    controller = PipelineController(
        vcf_path=Path(args.vcf),
        output_dir=Path(args.output_dir),
        config_path=Path(args.config),
        skip_stage2=args.skip_stage2,
        dry_run=args.dry_run,
    )
    try:
        summary = controller.run()
    except Exception:
        return 1

    if summary["status"] == "dry_run":
        print("Dry run completed successfully")
        print("Execution order:")
        for stage in summary["execution_order"]:
            print(f"  - {stage}")
        return 0

    print("Pipeline completed successfully")
    print("")
    print("Stages completed:")
    for stage in summary["completed_stages"]:
        print(f"  - {stage}")
    print("")
    print(f"Total runtime: {summary['total_runtime_seconds']:.2f} seconds")
    print(f"Final output: {summary['final_output']}")
    print(f"QC report: {summary['qc_report']}")
    print(f"Overall reliability score: {summary['overall_reliability_score']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

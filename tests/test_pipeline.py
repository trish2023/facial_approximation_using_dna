from __future__ import annotations

import gzip
import json
import tempfile
import unittest
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import cv2
import numpy as np

import run_pipeline


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.vcf_path = self.root / "subject.vcf.gz"
        self.output_dir = self.root / "outputs"
        self._write_minimal_vcf(self.vcf_path)

    def tearDown(self) -> None:
        self.tmpdir.cleanup()

    def _write_minimal_vcf(self, path: Path) -> None:
        content = """##fileformat=VCFv4.2
##source=unit-test
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tHG01583
1\t1000\trsTEST\tA\tG\t.\tPASS\t.\tGT\t0/1
"""
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(content)

    def _write_png(self, path: Path, size: tuple[int, int] = (16, 16), colour: tuple[int, int, int] = (0, 0, 0)) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((size[1], size[0], 3), colour, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), image))

    def _stage_side_effect(self, command: list[str], *_args, **_kwargs) -> CompletedProcess[str]:
        script_name = Path(command[1]).name
        project = run_pipeline.PROJECT_ROOT
        outputs = project / "outputs"
        processed = project / "data" / "processed"

        if script_name == "superpop_classifier.py":
            (outputs / "ancestry_stage1.json").write_text(
                json.dumps(
                    {
                        "pipeline": "KNN",
                        "predicted_superpopulation": "SAS",
                        "probabilities": {"SAS": 1.0, "EUR": 0.0},
                        "is_south_asian": True,
                        "subject_id": "HG01583",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "stage2_random_forest.py":
            (outputs / "ancestry_stage2.json").write_text(
                json.dumps(
                    {
                        "subject_id": "HG01583",
                        "predicted_label": "IndoAryan-proxy",
                        "probabilities": {"IndoAryan-proxy": 0.76, "Dravidian-proxy": 0.24},
                        "ani_proportion": 0.76,
                        "asi_proportion": 0.24,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "predict_sex.py":
            (outputs / "sex_prediction.json").write_text(
                json.dumps(
                    {
                        "predicted_sex": "male",
                        "confidence": 0.99,
                        "x_variant_count": None,
                        "y_variant_count": None,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "extract_hirisplex_snps.py":
            (outputs / "hirisplex_dosages.csv").write_text(
                "rsID,dosage\nrs1,1\n",
                encoding="utf-8",
            )
        elif script_name == "run_hirisplex.py":
            (outputs / "hirisplex_predictions.json").write_text(
                json.dumps(
                    {
                        "predictions": {
                            "eye": {"predicted": "brown", "probabilities": {"brown": 1.0}, "n_present": 32, "n_imputed": 9},
                            "hair": {"predicted": "black", "probabilities": {"black": 1.0}, "n_present": 32, "n_imputed": 9},
                            "skin": {"predicted": "pale", "probabilities": {"pale": 1.0}, "n_present": 32, "n_imputed": 9},
                        }
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "pigmentation_uncertainty.py":
            (outputs / "pigmentation_report.json").write_text(
                json.dumps(
                    {
                        "subject_id": "HG01583",
                        "snp_coverage": {"n_present": 32, "n_total": 41, "coverage_pct": 78.05},
                        "trait_reports": {
                            "eye": {"predicted": "brown", "max_probability": 0.99},
                            "hair": {"predicted": "black", "max_probability": 0.86},
                            "skin": {"predicted": "pale", "max_probability": 0.82},
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "weighted_prs.py":
            (processed / "prs_scores.json").write_text(
                json.dumps({"xiong2025_Chin_35_36": {"prs": -0.2144}}, indent=2),
                encoding="utf-8",
            )
            (processed / "prs_zscores.json").write_text(
                json.dumps({"xiong2025_Chin_35_36": {"z_score": -1.7888}}, indent=2),
                encoding="utf-8",
            )
            (processed / "prs_summary.csv").write_text(
                "trait,prs,z_score,snp_count,imputed_count,shared_snps,eur_only,du_only,alpha_used\n"
                "xiong2025_Chin_35_36,-0.2144,-1.7888,5,0,0,5,0,0.76\n",
                encoding="utf-8",
            )
        elif script_name == "normalise_prs.py":
            (processed / "morphology_traits.json").write_text(
                json.dumps(
                    {
                        "xiong2025_Chin_35_36": {
                            "z_score": -1.7888,
                            "direction": "Decreased",
                            "magnitude": "Moderate",
                            "confidence_score": 4.0,
                            "snp_count": 5,
                        }
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (processed / "top_traits.json").write_text("[]", encoding="utf-8")
        elif script_name == "template_selection.py":
            (outputs / "template_selection.json").write_text(
                json.dumps(
                    {
                        "template_path": "data/reference/face_templates/indoaryan_male.jpg",
                        "template_name": "indoaryan_male",
                        "ancestry_label": "IndoAryan-proxy",
                        "sex": "male",
                        "metadata": {},
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "landmark_detection.py":
            landmarks = [
                {"id": idx, "x": idx * 2, "y": idx * 2}
                for idx in range(76)
            ]
            (processed / "template_landmarks.json").write_text(
                json.dumps({"landmark_count": 76, "landmarks": landmarks}, indent=2),
                encoding="utf-8",
            )
            self._write_png(outputs / "template_landmarks_overlay.png")
        elif script_name == "landmark_displacement_mapping.py":
            (processed / "landmark_displacements.json").write_text(
                json.dumps({"landmark_8": {"dx": 0.0, "dy": -10.0}}, indent=2),
                encoding="utf-8",
            )
            (outputs / "displacement_summary.json").write_text(
                json.dumps({"traits_used": [], "landmarks_modified": [8], "max_displacement": 10.0, "mean_displacement": 10.0}, indent=2),
                encoding="utf-8",
            )
        elif script_name == "delaunay_warp.py":
            self._write_png(outputs / "morphology_warped_face.png")
            self._write_png(outputs / "warp_difference_heatmap.png")
            self._write_png(outputs / "warp_comparison.png")
            (outputs / "warp_qc.json").write_text(
                json.dumps(
                    {
                        "triangle_count": 140,
                        "landmark_count": 76,
                        "max_displacement_px": 15.3,
                        "mean_displacement_px": 1.5,
                        "p95_displacement_px": 11.8,
                        "unrealistic_warp": False,
                        "boundary_smoothing_applied": True,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "apply_pigmentation.py":
            self._write_png(outputs / "pigmentation_applied_face.png")
            (outputs / "pigmentation_qc.json").write_text(
                json.dumps(
                    {
                        "eye_colour": "brown",
                        "hair_colour": "black",
                        "skin_tone": "pale",
                        "watermark_added": True,
                        "eye_mask_area": 94,
                        "hair_mask_area": 128155,
                        "skin_mask_area": 69689,
                        "mean_colour_shift": 55.46,
                        "maximum_colour_shift": 151.88,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        elif script_name == "generate_qc_report.py":
            (outputs / "qc_report.json").write_text(
                json.dumps(
                    {
                        "overview": {
                            "overall_reliability": 0.773,
                            "overall_reliability_score": 0.773,
                            "traffic_light": "GREEN",
                        }
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            (outputs / "qc_report.md").write_text("# QC", encoding="utf-8")

        return CompletedProcess(command, 0, stdout="", stderr="")

    def test_dry_run_execution(self) -> None:
        out_dir = self.root / "dry_run_outputs"
        with patch("run_pipeline.subprocess.run") as mocked_run:
            controller = run_pipeline.PipelineController(
                vcf_path=self.vcf_path,
                output_dir=out_dir,
                dry_run=True,
            )
            summary = controller.run()
        self.assertEqual(summary["status"], "dry_run")
        self.assertGreater(len(summary["execution_order"]), 0)
        mocked_run.assert_not_called()
        self.assertFalse((out_dir / "qc_report.json").exists())

    def test_synthetic_minimal_vcf_sample_id(self) -> None:
        self.assertEqual(run_pipeline.read_vcf_sample_id(self.vcf_path), "HG01583")

    def test_successful_pipeline_execution(self) -> None:
        out_dir = self.root / "final_outputs"
        with patch("run_pipeline.subprocess.run", side_effect=self._stage_side_effect):
            controller = run_pipeline.PipelineController(
                vcf_path=self.vcf_path,
                output_dir=out_dir,
            )
            summary = controller.run()

        self.assertEqual(summary["status"], "success")
        self.assertTrue((out_dir / "pigmentation_applied_face.png").exists())
        self.assertTrue((out_dir / "qc_report.json").exists())
        self.assertTrue((out_dir / "prs_scores.json").exists())
        self.assertTrue((out_dir / "prs_zscores.json").exists())
        self.assertTrue((out_dir / "prs_summary.csv").exists())
        self.assertTrue((out_dir / "morphology_traits.json").exists())
        self.assertTrue((out_dir / "top_traits.json").exists())

        qc = json.loads((out_dir / "qc_report.json").read_text(encoding="utf-8"))
        self.assertIn("overall_reliability_score", qc["overview"])

    def test_generated_qc_report_contains_reliability_score(self) -> None:
        out_dir = self.root / "report_outputs"
        with patch("run_pipeline.subprocess.run", side_effect=self._stage_side_effect):
            controller = run_pipeline.PipelineController(
                vcf_path=self.vcf_path,
                output_dir=out_dir,
            )
            controller.run()

        qc = json.loads((out_dir / "qc_report.json").read_text(encoding="utf-8"))
        self.assertGreaterEqual(qc["overview"]["overall_reliability_score"], 0.0)

    def test_failure_writes_failed_stage_file(self) -> None:
        out_dir = self.root / "failed_outputs"

        def fail_on_landmarks(command: list[str], *_args, **_kwargs) -> CompletedProcess[str]:
            if Path(command[1]).name == "landmark_detection.py":
                return CompletedProcess(command, 2, stdout="", stderr="landmark detector unavailable")
            return self._stage_side_effect(command)

        with patch("run_pipeline.subprocess.run", side_effect=fail_on_landmarks):
            controller = run_pipeline.PipelineController(
                vcf_path=self.vcf_path,
                output_dir=out_dir,
            )
            with self.assertRaises(RuntimeError):
                controller.run()

        failed = json.loads((out_dir / "PIPELINE_FAILED.json").read_text(encoding="utf-8"))
        self.assertEqual(failed["failed_stage"], "landmark_detection")
        self.assertIn("landmark detector unavailable", failed["error"])


if __name__ == "__main__":
    unittest.main()

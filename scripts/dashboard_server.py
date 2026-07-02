#!/usr/bin/env python3
"""Local dashboard for running and reviewing the facial approximation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
RUN_STATE_PATH = OUTPUT_DIR / "dashboard_run_state.json"

DEFAULT_VCF_OPTIONS = [
    PROJECT_ROOT / "data" / "processed" / "1kg_SAS_EUR_merged.vcf.gz",
    PROJECT_ROOT / "data" / "processed" / "subject_hirisplex.vcf",
]

STAGE_DEFS = [
    {
        "key": "extract_subject_vcf",
        "title": "Subject VCF",
        "description": "Prepare a single-sample VCF for the pipeline.",
        "outputs": [PROJECT_ROOT / "data" / "processed" / "subject_hirisplex.vcf"],
    },
    {
        "key": "ancestry_superpopulation",
        "title": "Ancestry Stage 1",
        "description": "Predict broad super-population assignment.",
        "outputs": [PROJECT_ROOT / "outputs" / "ancestry_stage1.json"],
    },
    {
        "key": "ancestry_subclassification",
        "title": "Ancestry Stage 2",
        "description": "Refine South Asian sub-classification.",
        "outputs": [PROJECT_ROOT / "outputs" / "ancestry_stage2.json"],
    },
    {
        "key": "sex_prediction",
        "title": "Sex Prediction",
        "description": "Select the most suitable face template.",
        "outputs": [PROJECT_ROOT / "outputs" / "sex_prediction.json"],
    },
    {
        "key": "extract_hirisplex_dosages",
        "title": "HIrisPlex Extraction",
        "description": "Map the 41 HIrisPlex SNPs and compute dosages.",
        "outputs": [PROJECT_ROOT / "outputs" / "hirisplex_dosages.csv"],
    },
    {
        "key": "run_hirisplex_predictions",
        "title": "HIrisPlex Prediction",
        "description": "Predict eye, hair, and skin pigmentation.",
        "outputs": [PROJECT_ROOT / "outputs" / "hirisplex_predictions.json"],
    },
    {
        "key": "pigmentation_uncertainty",
        "title": "Pigmentation QC",
        "description": "Summarise confidence and SNP coverage.",
        "outputs": [PROJECT_ROOT / "outputs" / "pigmentation_report.json"],
    },
    {
        "key": "prs_computation",
        "title": "PRS Scores",
        "description": "Score morphology-related polygenic traits.",
        "outputs": [PROJECT_ROOT / "outputs" / "prs_summary.csv"],
    },
    {
        "key": "morphology_generation",
        "title": "Morphology",
        "description": "Normalise PRS values into trait directions.",
        "outputs": [PROJECT_ROOT / "outputs" / "morphology_traits.json"],
    },
    {
        "key": "template_selection",
        "title": "Template Selection",
        "description": "Choose the neutral base face.",
        "outputs": [PROJECT_ROOT / "outputs" / "template_selection.json"],
    },
    {
        "key": "landmark_detection",
        "title": "Landmark Detection",
        "description": "Detect the facial anchor points.",
        "outputs": [PROJECT_ROOT / "data" / "processed" / "template_landmarks.json"],
    },
    {
        "key": "displacement_mapping",
        "title": "Displacement Map",
        "description": "Convert morphology to landmark movement.",
        "outputs": [PROJECT_ROOT / "data" / "processed" / "landmark_displacements.json"],
    },
    {
        "key": "delaunay_warp",
        "title": "Warp",
        "description": "Warp the neutral face into the predicted shape.",
        "outputs": [PROJECT_ROOT / "outputs" / "morphology_warped_face.png", PROJECT_ROOT / "outputs" / "warp_qc.json"],
    },
    {
        "key": "apply_pigmentation",
        "title": "Pigmentation",
        "description": "Apply the predicted eye, hair, and skin colour.",
        "outputs": [PROJECT_ROOT / "outputs" / "pigmentation_applied_face.png", PROJECT_ROOT / "outputs" / "pigmentation_qc.json"],
    },
    {
        "key": "final_forensic_render",
        "title": "Final Composite",
        "description": "Build the annotation page and final report asset.",
        "outputs": [PROJECT_ROOT / "outputs" / "final_composite.png", PROJECT_ROOT / "outputs" / "final_composite.pdf", PROJECT_ROOT / "outputs" / "final_render_validation.json"],
    },
    {
        "key": "qc_report",
        "title": "QC Report",
        "description": "Aggregate the full audit trail.",
        "outputs": [PROJECT_ROOT / "outputs" / "qc_report.json", PROJECT_ROOT / "outputs" / "qc_report.md"],
    },
]


def read_text(path: Path, limit: int | None = None) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if limit is not None and len(text) > limit:
        return text[-limit:]
    return text


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def sample_id_from_vcf(path: Path) -> str | None:
    if not path.exists():
        return None
    opener = open
    if path.suffix == ".gz":
        import gzip

        opener = gzip.open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                return cols[9] if len(cols) > 9 else None
    return None


def is_process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def stage_status(stage_key: str, completed: list[str], current: str | None, outputs: list[Path], error_stage: str | None, running: bool) -> str:
    if error_stage == stage_key:
        return "failed"
    if running and current == stage_key:
        return "running"
    if stage_key in completed:
        return "complete"
    if outputs and all(path.exists() for path in outputs):
        return "complete"
    return "pending"


def verification_for_path(path: Path) -> dict[str, Any]:
    exists = path.exists()
    info: dict[str, Any] = {"path": str(path.relative_to(PROJECT_ROOT)), "exists": exists}
    if exists:
        stat = path.stat()
        info["size_kb"] = round(stat.st_size / 1024, 1)
        info["modified"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
    return info


def parse_log_blocks(log_text: str) -> dict[str, list[str]]:
    blocks: dict[str, list[str]] = {}
    current: str | None = None
    for raw_line in log_text.splitlines():
        line = raw_line.rstrip()
        if "Starting stage:" in line:
            current = line.split("Starting stage:", 1)[1].strip()
            blocks.setdefault(current, []).append(line)
            continue
        if current:
            blocks[current].append(line)
            if f"Completed stage: {current}" in line or f"failed in stage {current}" in line.lower():
                current = None
    return blocks


def summarize_state() -> dict[str, Any]:
    state = load_json(RUN_STATE_PATH) or {}
    running = is_process_alive(state.get("pid"))
    log_text = read_text(OUTPUT_DIR / "pipeline.log", limit=30000)
    blocks = parse_log_blocks(log_text)
    completed = state.get("completed_stages", [])
    current = state.get("current_stage")
    return {
        "run": {
            "running": running,
            "pid": state.get("pid"),
            "source_vcf": state.get("source_vcf"),
            "sample_id": state.get("sample_id"),
            "started_at": state.get("started_at"),
            "current_stage": current,
            "completed_stages": completed,
            "error": state.get("error"),
        },
        "summary": {
            "subject_id": state.get("subject_id"),
            "final_composite": verification_for_path(OUTPUT_DIR / "final_composite.png"),
            "qc_report": verification_for_path(OUTPUT_DIR / "qc_report.json"),
            "overall_reliability": (load_json(OUTPUT_DIR / "qc_report.json") or {}).get("overview", {}).get("overall_reliability"),
        },
        "input_options": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "label": path.name,
                "sample_id": sample_id_from_vcf(path),
                "exists": path.exists(),
            }
            for path in DEFAULT_VCF_OPTIONS
        ],
        "stages": [
            {
                **stage,
                "status": stage_status(
                    stage["key"],
                    completed if isinstance(completed, list) else [],
                    current if isinstance(current, str) else None,
                    stage["outputs"],
                    state.get("error_stage") if isinstance(state, dict) else None,
                    running,
                ),
                "outputs": [verification_for_path(path) for path in stage["outputs"]],
                "log_excerpt": "\n".join(blocks.get(stage["key"], [])[-80:])[-4000:],
                "verification": stage_verification(stage["key"]),
            }
            for stage in STAGE_DEFS
        ],
        "log_tail": log_text[-12000:],
    }


def stage_verification(stage_key: str) -> list[dict[str, Any]]:
    mapping = {
        "extract_subject_vcf": [verification_for_path(PROJECT_ROOT / "data" / "processed" / "subject_hirisplex.vcf")],
        "ancestry_superpopulation": [verification_for_path(OUTPUT_DIR / "ancestry_stage1.json")],
        "ancestry_subclassification": [verification_for_path(OUTPUT_DIR / "ancestry_stage2.json")],
        "sex_prediction": [verification_for_path(OUTPUT_DIR / "sex_prediction.json")],
        "extract_hirisplex_dosages": [verification_for_path(OUTPUT_DIR / "hirisplex_dosages.csv")],
        "run_hirisplex_predictions": [verification_for_path(OUTPUT_DIR / "hirisplex_predictions.json")],
        "pigmentation_uncertainty": [verification_for_path(OUTPUT_DIR / "pigmentation_report.json"), verification_for_path(OUTPUT_DIR / "pigmentation_qc.json")],
        "prs_computation": [verification_for_path(OUTPUT_DIR / "prs_summary.csv"), verification_for_path(PROJECT_ROOT / "data" / "processed" / "prs_scores.json")],
        "morphology_generation": [verification_for_path(OUTPUT_DIR / "morphology_traits.json"), verification_for_path(OUTPUT_DIR / "top_traits.json")],
        "template_selection": [verification_for_path(OUTPUT_DIR / "template_selection.json")],
        "landmark_detection": [verification_for_path(PROJECT_ROOT / "data" / "processed" / "template_landmarks.json"), verification_for_path(OUTPUT_DIR / "template_landmarks_overlay.png")],
        "displacement_mapping": [verification_for_path(PROJECT_ROOT / "data" / "processed" / "landmark_displacements.json"), verification_for_path(OUTPUT_DIR / "displacement_summary.json")],
        "delaunay_warp": [verification_for_path(OUTPUT_DIR / "morphology_warped_face.png"), verification_for_path(OUTPUT_DIR / "warp_qc.json")],
        "apply_pigmentation": [verification_for_path(OUTPUT_DIR / "pigmentation_applied_face.png"), verification_for_path(OUTPUT_DIR / "pigmentation_qc.json")],
        "final_forensic_render": [verification_for_path(OUTPUT_DIR / "final_composite.png"), verification_for_path(OUTPUT_DIR / "final_composite.pdf"), verification_for_path(OUTPUT_DIR / "final_render_validation.json")],
        "qc_report": [verification_for_path(OUTPUT_DIR / "qc_report.json"), verification_for_path(OUTPUT_DIR / "qc_report.md")],
    }
    return mapping.get(stage_key, [])


def json_response(handler: SimpleHTTPRequestHandler, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
    data = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def send_local_file(handler: SimpleHTTPRequestHandler, path: Path) -> bool:
    if not path.exists() or not path.is_file():
        return False
    data = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    if path.suffix == ".png":
        content_type = "image/png"
    elif path.suffix == ".jpg" or path.suffix == ".jpeg":
        content_type = "image/jpeg"
    elif path.suffix == ".pdf":
        content_type = "application/pdf"
    elif path.suffix == ".json":
        content_type = "application/json; charset=utf-8"
    elif path.suffix == ".csv":
        content_type = "text/csv; charset=utf-8"
    elif path.suffix == ".md":
        content_type = "text/markdown; charset=utf-8"
    else:
        content_type = "application/octet-stream"
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return True


def start_pipeline(body: dict[str, Any]) -> dict[str, Any]:
    vcf = Path(body.get("vcf") or PROJECT_ROOT / "data" / "processed" / "1kg_SAS_EUR_merged.vcf.gz")
    sample = body.get("sample")
    skip_stage2 = bool(body.get("skip_stage2", False))
    dry_run = bool(body.get("dry_run", False))
    output_dir = Path(body.get("output_dir") or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_pipeline.py"),
        "--vcf",
        str(vcf),
        "--output-dir",
        str(output_dir),
    ]
    if sample:
        cmd.extend(["--sample", str(sample)])
    if skip_stage2:
        cmd.append("--skip-stage2")
    if dry_run:
        cmd.append("--dry-run")

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    state = {
        "pid": proc.pid,
        "source_vcf": str(vcf.relative_to(PROJECT_ROOT)) if vcf.is_relative_to(PROJECT_ROOT) else str(vcf),
        "sample_id": sample,
        "output_dir": str(output_dir),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "current_stage": None,
        "completed_stages": [],
        "subject_id": sample,
        "error": None,
    }
    RUN_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return {"ok": True, "pid": proc.pid, "command": cmd}


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/state":
            json_response(self, summarize_state())
            return
        if parsed.path == "/api/options":
            json_response(
                self,
                {
                    "default_vcf": str((PROJECT_ROOT / "data" / "processed" / "1kg_SAS_EUR_merged.vcf.gz").relative_to(PROJECT_ROOT)),
                    "vcfs": [
                        {
                            "path": str(path.relative_to(PROJECT_ROOT)),
                            "label": path.name,
                            "sample_id": sample_id_from_vcf(path),
                            "exists": path.exists(),
                        }
                        for path in DEFAULT_VCF_OPTIONS
                    ],
                },
            )
            return
        if parsed.path.startswith("/api/log"):
            json_response(self, {"tail": (OUTPUT_DIR / "pipeline.log").read_text(encoding="utf-8", errors="ignore")[-12000:] if (OUTPUT_DIR / "pipeline.log").exists() else ""})
            return
        if parsed.path.startswith("/outputs/") or parsed.path.startswith("/data/"):
            rel = parsed.path.lstrip("/")
            target = PROJECT_ROOT / rel
            if send_local_file(self, target):
                return
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/run":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body = json.loads(raw) if raw else {}
            if not isinstance(body, dict):
                raise ValueError("Body must be a JSON object")
        except Exception as exc:  # pragma: no cover - defensive
            json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        try:
            payload = start_pipeline(body)
        except Exception as exc:  # pragma: no cover - defensive
            json_response(self, {"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        json_response(self, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local dashboard server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Dashboard running on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

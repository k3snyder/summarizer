#!/usr/bin/env python3
"""
Submit a document to the summarizer backend, poll until completion,
download JSON/Markdown outputs, and print a manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_CONFIG: dict[str, Any] = {
    "extract_only": False,
    "skip_tables": False,
    "skip_images": False,
    "skip_pptx_tables": False,
    "text_only": False,
    "pdf_image_dpi": 200,
    "vision_mode": "llama_cpp",
    "vision_classifier_mode": None,
    "vision_extractor_mode": None,
    "vision_cli_provider": None,
    "vision_detailed_extraction": False,
    "chunk_size": 3000,
    "chunk_overlap": 80,
    "run_summarization": True,
    "summarizer_mode": "full",
    "summarizer_provider": "llama_cpp",
    "summarizer_cli_provider": None,
    "summarizer_detailed_extraction": False,
    "summarizer_insight_mode": False,
    "keep_base64_images": False,
}

VISION_CHOICES = ["none", "deepseek", "gemini", "openai", "ollama", "llama_cpp", "codex", "claude"]
VISION_SUBMODE_CHOICES = ["none", "ollama", "llama_cpp", "openai", "gemini", "codex", "claude"]
SUMMARIZER_CHOICES = ["ollama", "llama_cpp", "openai", "codex", "claude"]
CLI_CHOICES = ["codex", "claude"]
TERMINAL_STATUSES = {"completed", "failed"}


class JobClientError(RuntimeError):
    """Raised for expected client-side failures."""


def eprint(message: str) -> None:
    print(message, file=sys.stderr)


def normalized_backend_url(raw_url: str | None) -> str:
    base_url = (
        raw_url
        or os.environ.get("SUMMARIZER_BACKEND_URL")
        or os.environ.get("NEXT_PUBLIC_BACKEND_URL")
        or "http://localhost:8000"
    )
    return base_url.rstrip("/")


def load_json_response(url: str, timeout_seconds: float) -> dict[str, Any]:
    req = request.Request(
        url,
        method="GET",
        headers={"Accept": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = extract_detail(body)
        raise JobClientError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except error.URLError as exc:
        raise JobClientError(f"Unable to reach {url}: {exc.reason}") from exc


def download_file(url: str, destination: Path, timeout_seconds: float, accept: str) -> None:
    req = request.Request(
        url,
        method="GET",
        headers={"Accept": accept},
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            destination.write_bytes(response.read())
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        detail = extract_detail(body)
        raise JobClientError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except error.URLError as exc:
        raise JobClientError(f"Unable to reach {url}: {exc.reason}") from exc


def extract_detail(raw_body: str) -> str:
    raw_body = raw_body.strip()
    if not raw_body:
        return "(empty response)"
    try:
        parsed = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body
    if isinstance(parsed, dict) and isinstance(parsed.get("detail"), str):
        return parsed["detail"]
    return raw_body


def submit_job(file_path: Path, config: dict[str, Any], backend_url: str, timeout_seconds: float) -> dict[str, Any]:
    curl_path = shutil.which("curl")
    if not curl_path:
        raise JobClientError("curl is required for streaming multipart uploads but was not found in PATH")

    config_json = json.dumps(config, separators=(",", ":"), ensure_ascii=True)
    endpoint = f"{backend_url}/api/jobs"
    command = [
        curl_path,
        "--silent",
        "--show-error",
        "--fail-with-body",
        "-X",
        "POST",
        endpoint,
        "-F",
        f"file=@{file_path}",
        "-F",
        f"config={config_json}",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        detail = extract_detail((completed.stdout or completed.stderr or "").strip())
        raise JobClientError(f"Job submission failed: {detail}")

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise JobClientError(f"Could not parse job creation response: {completed.stdout}") from exc


def poll_job(job_id: str, backend_url: str, poll_interval: float, timeout_seconds: float) -> dict[str, Any]:
    status_url = f"{backend_url}/api/jobs/{job_id}"
    started = time.time()
    last_line = None

    while True:
        if time.time() - started > timeout_seconds:
            raise JobClientError(f"Timed out waiting for job {job_id} after {timeout_seconds} seconds")

        status_data = load_json_response(status_url, timeout_seconds)
        line = (
            f"[{status_data.get('status')}] "
            f"{status_data.get('progress', 0)}% "
            f"{status_data.get('current_stage') or '-'} "
            f"{status_data.get('message') or ''}"
        ).strip()
        if line != last_line:
            eprint(line)
            last_line = line

        if status_data.get("status") in TERMINAL_STATUSES:
            return status_data

        time.sleep(poll_interval)


def timestamped_output_dir(base_dir: Path, file_path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_name = file_path.stem.replace(" ", "-")
    return base_dir / f"{safe_name}-{stamp}"


def parse_key_value(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"Invalid override '{item}'. Use key=value.")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"Invalid override '{item}'. Key is empty.")
    return key, parse_scalar(value.strip())


def parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"null", "none"}:
        return None
    if value.startswith("{") or value.startswith("[") or value.startswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    try:
        return int(value)
    except ValueError:
        return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a document to the summarizer backend and download outputs.")
    parser.add_argument("--file", required=True, help="Path to PDF, PPTX, TXT, or MD document")
    parser.add_argument("--backend-url", help="Backend base URL (default: SUMMARIZER_BACKEND_URL, NEXT_PUBLIC_BACKEND_URL, or http://localhost:8000)")
    parser.add_argument("--output-dir", help="Directory to write output artifacts into")
    parser.add_argument("--config-json", help="JSON object to merge onto the default pipeline config")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Arbitrary config override (repeatable)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")
    parser.add_argument("--timeout-seconds", type=float, default=3600.0, help="Total timeout for submission + polling")

    parser.add_argument("--extract-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-tables", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-images", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--skip-pptx-tables", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--text-only", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pdf-image-dpi", type=int, choices=[72, 144, 200, 300], default=None)
    parser.add_argument("--vision-mode", choices=VISION_CHOICES, default=None)
    parser.add_argument("--vision-classifier-mode", choices=VISION_SUBMODE_CHOICES, default=None)
    parser.add_argument("--vision-extractor-mode", choices=VISION_SUBMODE_CHOICES, default=None)
    parser.add_argument("--vision-cli-provider", choices=CLI_CHOICES, default=None)
    parser.add_argument("--vision-detailed-extraction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--run-summarization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--summarizer-mode", choices=["full", "topics-only", "skip"], default=None)
    parser.add_argument("--summarizer-provider", choices=SUMMARIZER_CHOICES, default=None)
    parser.add_argument("--summarizer-cli-provider", choices=CLI_CHOICES, default=None)
    parser.add_argument("--summarizer-detailed-extraction", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--summarizer-insight-mode", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--keep-base64-images", action=argparse.BooleanOptionalAction, default=None)
    return parser


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)

    if args.config_json:
        try:
            loaded = json.loads(args.config_json)
        except json.JSONDecodeError as exc:
            raise JobClientError(f"--config-json is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise JobClientError("--config-json must decode to a JSON object")
        config.update(loaded)

    explicit_values = {
        "extract_only": args.extract_only,
        "skip_tables": args.skip_tables,
        "skip_images": args.skip_images,
        "skip_pptx_tables": args.skip_pptx_tables,
        "text_only": args.text_only,
        "pdf_image_dpi": args.pdf_image_dpi,
        "vision_mode": args.vision_mode,
        "vision_classifier_mode": args.vision_classifier_mode,
        "vision_extractor_mode": args.vision_extractor_mode,
        "vision_cli_provider": args.vision_cli_provider,
        "vision_detailed_extraction": args.vision_detailed_extraction,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "run_summarization": args.run_summarization,
        "summarizer_mode": args.summarizer_mode,
        "summarizer_provider": args.summarizer_provider,
        "summarizer_cli_provider": args.summarizer_cli_provider,
        "summarizer_detailed_extraction": args.summarizer_detailed_extraction,
        "summarizer_insight_mode": args.summarizer_insight_mode,
        "keep_base64_images": args.keep_base64_images,
    }
    for key, value in explicit_values.items():
        if value is not None:
            config[key] = value

    for item in args.set:
        key, value = parse_key_value(item)
        config[key] = value

    return config


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    file_path = Path(args.file).expanduser().resolve()
    if not file_path.exists():
        raise JobClientError(f"File not found: {file_path}")
    if not file_path.is_file():
        raise JobClientError(f"Path is not a file: {file_path}")

    backend_url = normalized_backend_url(args.backend_url)
    load_json_response(f"{backend_url}/api/health", timeout_seconds=10.0)

    output_base = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path.cwd() / ".summarizer-cli-runs"
    output_dir = timestamped_output_dir(output_base, file_path) if args.output_dir is None else output_base
    output_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args)
    created = submit_job(file_path, config, backend_url, timeout_seconds=min(args.timeout_seconds, 300.0))
    job_id = created.get("job_id")
    if not isinstance(job_id, str) or not job_id:
        raise JobClientError(f"Unexpected job creation response: {created}")

    eprint(f"job_id={job_id}")
    final_status = poll_job(job_id, backend_url, args.poll_interval, args.timeout_seconds)

    job_meta_path = output_dir / "job-meta.json"
    output_json_path = output_dir / "output.json"
    output_md_path = output_dir / "output.md"

    manifest: dict[str, Any] = {
        "job_id": job_id,
        "backend_url": backend_url,
        "output_dir": str(output_dir),
        "job_meta_path": str(job_meta_path),
        "config": config,
        "created_response": created,
        "final_status": final_status,
    }

    if final_status.get("status") != "completed":
        job_meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        raise JobClientError(f"Job {job_id} failed: {final_status.get('error') or final_status.get('message') or 'unknown error'}")

    download_file(f"{backend_url}/api/jobs/{job_id}/output", output_json_path, args.timeout_seconds, "application/json")
    raw_output = json.loads(output_json_path.read_text(encoding="utf-8"))
    output_json_path.write_text(json.dumps(raw_output, indent=2, sort_keys=True), encoding="utf-8")

    download_file(f"{backend_url}/api/jobs/{job_id}/output/markdown", output_md_path, args.timeout_seconds, "text/markdown")

    manifest.update(
        {
            "status": "completed",
            "output_json_path": str(output_json_path),
            "output_markdown_path": str(output_md_path),
        }
    )
    job_meta_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JobClientError as exc:
        eprint(f"ERROR: {exc}")
        raise SystemExit(1)

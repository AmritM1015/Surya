#!/usr/bin/env python3
"""Standalone Surya easy inference: prompt dates -> download -> rollout -> prediction.nc."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import h5netcdf
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader

from surya.datasets.helio import HelioNetCDFDataset, inverse_transform_single_channel
from surya.models.helio_spectformer import HelioSpectFormer
from surya.utils.data import build_scalers, custom_collate_fn

S3_FILE_PATTERN = re.compile(r"(\d{8})_(\d{4})\.nc$")
DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")

DEFAULT_USER = {
    "start_datetime": "2014-10-23 10:00:00",
    "end_datetime": "2014-10-23 17:00:00",
    "prompt_for_dates": True,
    "output_dir": "easy_inference/outputs_20141023_60min",
    "samples": 1,
}

DEFAULT_ADVANCED = {
    "foundation_config_path": "data/Surya-1.0/config.yaml",
    "scalers_path": "data/Surya-1.0/scalers.yaml",
    "weights_path": "data/Surya-1.0/surya.366m.v1.pt",
    "model_repo_id": "nasa-ibm-ai4science/Surya-1.0",
    "model_allow_patterns": ["config.yaml", "scalers.yaml", "surya.366m.v1.pt"],
    "validation_data_dir": "data/Surya-1.0_validation_data_20141023_60min",
    "index_path": "easy_inference/index_20141023_60min.csv",
    "cadence_minutes": 60,
    "time_delta_input_minutes": [-60, 0],
    "time_delta_target_minutes": 60,
    "rollout_steps": 2,
    "dataset_rollout_steps": 2,
    "s3_bucket": "nasa-surya-bench",
    "download_skip_existing": True,
    "download_verify_size": False,
    "download_match_tolerance_minutes": 0,
    "prune_validation_data_to_window": True,
    "device": "auto",
    "dtype": "auto",
    "batch_size": 1,
    "num_workers": 0,
    "prefetch_factor": 2,
    "warmup": 0,
    "disable_autocast": False,
    "cpu_threads": 0,
    "show_progress": True,
    "prediction_dtype": "float32",
}


@dataclass
class DownloadSummary:
    requested_timestamps: int
    matched_timestamps: int
    missing_timestamps: int
    pruned_files: int
    downloaded_files: int
    skipped_files: int
    failed_files: int
    output_dir: str


@dataclass
class InferenceSummary:
    avg_loss: float
    timed_batches: int
    avg_data_seconds: float
    avg_infer_seconds: float
    prediction_nc_path: str
    samples_written: int


class PredictionNetCDFWriter:
    def __init__(
        self,
        output_path: str,
        channels: list[str],
        prediction_dtype: np.dtype,
        input_steps: int,
        prediction_steps: int,
        shape_hw: tuple[int, int],
        sample_capacity: int,
    ) -> None:
        self.output_path = str(Path(output_path).resolve())
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)
        if Path(self.output_path).exists():
            Path(self.output_path).unlink()

        self.channels = channels
        self.input_steps = int(input_steps)
        self.prediction_steps = int(prediction_steps)
        self.height = int(shape_hw[0])
        self.width = int(shape_hw[1])
        self.sample_capacity = int(sample_capacity)
        self.prediction_dtype = np.dtype(prediction_dtype)
        self.ts_width = 32
        self.channel_width = max(8, max(len(ch) for ch in channels))

        self.file = h5netcdf.File(self.output_path, "w")
        self.file.dimensions = {
            "sample": self.sample_capacity,
            "prediction_time": self.prediction_steps,
            "input_time": self.input_steps,
            "y": self.height,
            "x": self.width,
            "channel": len(self.channels),
            "timestamp_strlen": self.ts_width,
            "channel_strlen": self.channel_width,
        }

        self.file.attrs["title"] = "Surya predictions"
        self.file.attrs["data_layout"] = "channel vars with dims (sample,prediction_time,y,x)"
        self.file.attrs["inverse_transform"] = "signum-log inverse applied"
        self.file.attrs["spatial_shape"] = f"{self.height}x{self.width}"
        self.file.attrs["prediction_dtype"] = self.prediction_dtype.name

        self.sample_id = self.file.create_variable("sample_id", ("sample",), dtype="i4")
        self.input_timestamps = self.file.create_variable(
            "input_timestamps", ("sample", "input_time", "timestamp_strlen"), dtype="S1"
        )
        self.prediction_timestamps = self.file.create_variable(
            "prediction_timestamps",
            ("sample", "prediction_time", "timestamp_strlen"),
            dtype="S1",
        )
        self.channel_names = self.file.create_variable(
            "channel_names", ("channel", "channel_strlen"), dtype="S1"
        )
        self.channel_names[...] = _encode_fixed_width(self.channels, self.channel_width)

        self.prediction_vars: dict[str, Any] = {}
        for channel in self.channels:
            self.prediction_vars[channel] = self.file.create_variable(
                channel,
                ("sample", "prediction_time", "y", "x"),
                dtype=self.prediction_dtype,
            )

    def write_sample_metadata(
        self,
        sample_idx: int,
        sample_id: int,
        timestamps_input,
        timestamps_prediction,
    ) -> None:
        input_strings = _datetime_strings(timestamps_input)
        prediction_strings = _datetime_strings(timestamps_prediction)
        self.sample_id[sample_idx] = int(sample_id)
        self.input_timestamps[sample_idx, :, :] = _encode_fixed_width(
            input_strings, self.ts_width
        )
        self.prediction_timestamps[sample_idx, :, :] = _encode_fixed_width(
            prediction_strings, self.ts_width
        )

    def write_prediction_frame(
        self,
        sample_idx: int,
        prediction_step_idx: int,
        channel_name: str,
        frame_hw: np.ndarray,
    ) -> None:
        self.prediction_vars[channel_name][sample_idx, prediction_step_idx, :, :] = frame_hw

    def finalize(self, samples_written: int) -> str:
        self.file.attrs["samples_written"] = int(samples_written)
        self.file.close()
        return self.output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Standalone easy Surya inference")
    parser.add_argument(
        "--config-path",
        default="easy_inference/config_easy.yaml",
        help="Path to easy YAML config.",
    )
    parser.add_argument(
        "--start-datetime",
        default=None,
        help="Override start datetime (UTC), format: YYYY-MM-DD HH:MM[:SS].",
    )
    parser.add_argument(
        "--end-datetime",
        default=None,
        help="Override end datetime (UTC), format: YYYY-MM-DD HH:MM[:SS].",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive date prompt.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip S3 download and reuse existing data files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved settings only.",
    )
    return parser.parse_args()


def _resolve_path(path_string: str, config_dir: Path) -> Path:
    _ = config_dir  # Keep signature stable; relative paths resolve from current working directory.
    path_obj = Path(path_string).expanduser()
    if path_obj.is_absolute():
        return path_obj.resolve()
    return (Path.cwd() / path_obj).resolve()


def _parse_datetime(value: str) -> datetime:
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError("Expected datetime format: YYYY-MM-DD HH:MM[:SS].")


def _format_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _prompt_datetime(label: str, default_value: datetime) -> datetime:
    default_text = _format_datetime(default_value)
    while True:
        value = input(f"{label} [{default_text}]: ").strip()
        if value == "":
            return default_value
        try:
            return _parse_datetime(value)
        except ValueError as exc:
            print(f"Invalid datetime '{value}': {exc}")


def _load_easy_sections(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(config_path, "r") as fp:
        raw = yaml.safe_load(fp) or {}
    user_cfg = deepcopy(DEFAULT_USER)
    user_cfg.update(raw.get("user", {}))
    advanced_cfg = deepcopy(DEFAULT_ADVANCED)
    advanced_cfg.update(raw.get("advanced", {}))
    return user_cfg, advanced_cfg


def _select_dates(
    user_cfg: dict[str, Any],
    cli_start: str | None,
    cli_end: str | None,
    use_prompt: bool,
) -> tuple[datetime, datetime]:
    start_dt = _parse_datetime(cli_start) if cli_start else _parse_datetime(user_cfg["start_datetime"])
    end_dt = _parse_datetime(cli_end) if cli_end else _parse_datetime(user_cfg["end_datetime"])
    if use_prompt:
        print("\nEnter download window in UTC (press Enter to keep defaults).")
        start_dt = _prompt_datetime("Start datetime", start_dt)
        end_dt = _prompt_datetime("End datetime", end_dt)
    if end_dt < start_dt:
        raise ValueError(
            f"End datetime must be >= start datetime. Got {_format_datetime(start_dt)} -> {_format_datetime(end_dt)}."
        )
    return start_dt, end_dt


def log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[progress] {message}", flush=True)


def _parse_timestamp_from_filename(filename: str) -> datetime | None:
    match = S3_FILE_PATTERN.search(filename)
    if not match:
        return None
    try:
        return datetime.strptime(f"{match.group(1)}_{match.group(2)}", "%Y%m%d_%H%M")
    except ValueError:
        return None


def _ensure_aws_cli_available() -> None:
    if shutil.which("aws") is None:
        raise RuntimeError(
            "AWS CLI is required for data download (`aws s3 cp ...`).\n"
            "Install it with uv, then retry:\n"
            "  uv add awscli\n"
            "  uv sync\n"
            "Alternative (current env only):\n"
            "  uv pip install awscli"
        )


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _list_s3_files_in_month(bucket: str, year: int, month: int) -> list[dict[str, Any]]:
    prefix = f"{year}/{month:02d}/"
    command = ["aws", "s3", "ls", f"s3://{bucket}/{prefix}", "--no-sign-request"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Failed listing s3://{bucket}/{prefix}: {stderr}")

    files: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        filename = parts[-1]
        if not filename.endswith(".nc"):
            continue
        timestamp = _parse_timestamp_from_filename(filename)
        if timestamp is None:
            continue
        size = int(parts[2]) if parts[2].isdigit() else 0
        files.append(
            {
                "path": f"{prefix}{filename}",
                "filename": filename,
                "size": size,
                "timestamp": timestamp,
            }
        )
    return files


def _expected_timestamps(start_datetime: datetime, end_datetime: datetime, cadence_minutes: int) -> list[datetime]:
    if cadence_minutes <= 0:
        raise ValueError("cadence_minutes must be positive.")
    cadence = timedelta(minutes=int(cadence_minutes))
    values = []
    curr = start_datetime
    while curr <= end_datetime:
        values.append(curr)
        curr += cadence
    return values


def _expected_filenames(start_datetime: datetime, end_datetime: datetime, cadence_minutes: int) -> set[str]:
    return {ts.strftime("%Y%m%d_%H%M.nc") for ts in _expected_timestamps(start_datetime, end_datetime, cadence_minutes)}


def prune_validation_dir_to_expected(output_dir: Path, expected_filenames: set[str], show_progress: bool) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in output_dir.glob("*.nc"):
        if path.name not in expected_filenames:
            path.unlink(missing_ok=True)
            removed += 1
    for path in output_dir.glob("*.nc.*"):
        path.unlink(missing_ok=True)
        removed += 1
    log_progress(show_progress, f"pruned validation dir | removed_files={removed}")
    return removed


def download_surya_bench_range(
    bucket: str,
    output_dir: Path,
    start_datetime: datetime,
    end_datetime: datetime,
    cadence_minutes: int,
    skip_existing: bool,
    verify_size: bool,
    match_tolerance_minutes: int,
    prune_to_expected: bool,
    show_progress: bool,
) -> DownloadSummary:
    _ensure_aws_cli_available()
    output_dir.mkdir(parents=True, exist_ok=True)

    expected = _expected_timestamps(start_datetime, end_datetime, cadence_minutes)
    expected_filenames = _expected_filenames(start_datetime, end_datetime, cadence_minutes)
    pruned_files = 0
    if prune_to_expected:
        pruned_files = prune_validation_dir_to_expected(
            output_dir=output_dir,
            expected_filenames=expected_filenames,
            show_progress=show_progress,
        )

    unique_months = sorted({(ts.year, ts.month) for ts in expected})
    available_files: list[dict[str, Any]] = []
    for year, month in unique_months:
        log_progress(show_progress, f"listing s3://{bucket}/{year}/{month:02d}/")
        available_files.extend(_list_s3_files_in_month(bucket=bucket, year=year, month=month))

    tolerance_minutes = int(match_tolerance_minutes)
    if tolerance_minutes < 0:
        raise ValueError("download_match_tolerance_minutes must be >= 0")
    range_start = start_datetime - timedelta(minutes=tolerance_minutes)
    range_end = end_datetime + timedelta(minutes=tolerance_minutes)
    window_files = [f for f in available_files if range_start <= f["timestamp"] <= range_end]

    matched_files: dict[datetime, dict[str, Any]] = {}
    for ts in expected:
        best_match = None
        best_diff = float("inf")
        for file_info in window_files:
            diff = abs((file_info["timestamp"] - ts).total_seconds() / 60)
            if diff <= tolerance_minutes and diff < best_diff:
                best_diff = diff
                best_match = file_info
        if best_match is not None:
            matched_files[ts] = best_match

    missing_timestamps = [ts for ts in expected if ts not in matched_files]
    log_progress(
        show_progress,
        (
            "download plan | "
            f"expected={len(expected)} matched={len(matched_files)} "
            f"missing={len(missing_timestamps)} tolerance_min={tolerance_minutes}"
        ),
    )

    if not matched_files:
        raise RuntimeError("No matching files found for requested date range.")

    downloaded = 0
    skipped = 0
    failed = 0
    ordered_timestamps = sorted(matched_files.keys())
    total = len(ordered_timestamps)
    for idx, ts in enumerate(ordered_timestamps, start=1):
        file_info = matched_files[ts]
        local_path = output_dir / f"{ts.strftime('%Y%m%d_%H%M')}.nc"

        if skip_existing and local_path.exists():
            if not verify_size:
                skipped += 1
                log_progress(show_progress, f"[{idx}/{total}] skip existing {local_path.name}")
                continue
            expected_size = int(file_info["size"])
            if expected_size > 0 and abs(local_path.stat().st_size - expected_size) <= 10240:
                skipped += 1
                log_progress(show_progress, f"[{idx}/{total}] skip existing {local_path.name} (size match)")
                continue

        log_progress(show_progress, f"[{idx}/{total}] download {local_path.name}")
        command = [
            "aws",
            "s3",
            "cp",
            f"s3://{bucket}/{file_info['path']}",
            str(local_path),
            "--no-sign-request",
            "--only-show-errors",
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
        if result.returncode == 0 and local_path.exists():
            downloaded += 1
            log_progress(show_progress, f"[{idx}/{total}] downloaded {local_path.name}")
        else:
            failed += 1
            stderr = (result.stderr or result.stdout).strip()
            if stderr:
                print(f"[download-error] {local_path.name}: {stderr}", file=sys.stderr)
            log_progress(show_progress, f"[{idx}/{total}] failed {local_path.name}")

    return DownloadSummary(
        requested_timestamps=len(expected),
        matched_timestamps=len(matched_files),
        missing_timestamps=len(missing_timestamps),
        pruned_files=pruned_files,
        downloaded_files=downloaded,
        skipped_files=skipped,
        failed_files=failed,
        output_dir=str(output_dir.resolve()),
    )


def build_index_csv_for_range(
    validation_data_dir: Path,
    index_path: Path,
    start_datetime: datetime,
    end_datetime: datetime,
    cadence_minutes: int,
) -> None:
    cadence = timedelta(minutes=int(cadence_minutes))
    rows = []
    curr = start_datetime
    while curr <= end_datetime:
        file_name = curr.strftime("%Y%m%d_%H%M.nc")
        abs_path = validation_data_dir / file_name
        rows.append(
            {
                "path": os.path.relpath(abs_path, Path.cwd()),
                "timestep": curr.strftime("%Y-%m-%d %H:%M:%S"),
                "present": 1 if abs_path.exists() else 0,
            }
        )
        curr += cadence
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(index_path, index=False)


def ensure_model_assets(advanced_cfg: dict[str, Any], config_dir: Path) -> None:
    foundation_config_path = _resolve_path(str(advanced_cfg["foundation_config_path"]), config_dir)
    scalers_path = _resolve_path(str(advanced_cfg["scalers_path"]), config_dir)
    weights_path = _resolve_path(str(advanced_cfg["weights_path"]), config_dir)
    if foundation_config_path.exists() and scalers_path.exists() and weights_path.exists():
        return

    model_dir = weights_path.parent
    model_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=str(advanced_cfg["model_repo_id"]),
        local_dir=str(model_dir),
        allow_patterns=list(advanced_cfg["model_allow_patterns"]),
        token=None,
    )


def resolve_device(device_arg: str) -> tuple[torch.device, str]:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda"), "cuda"
    if device_arg == "mps":
        if not _mps_available():
            raise RuntimeError("MPS requested but unavailable.")
        return torch.device("mps"), "mps"
    if device_arg == "cpu":
        return torch.device("cpu"), "cpu"
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda"
    if _mps_available():
        return torch.device("mps"), "mps"
    return torch.device("cpu"), "cpu"


def resolve_dtype(dtype_arg: str, device_type: str) -> torch.dtype:
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device_type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def supports_autocast(device_type: str, dtype: torch.dtype) -> bool:
    if device_type == "cuda":
        return dtype in (torch.float16, torch.bfloat16)
    if device_type == "cpu":
        return dtype == torch.bfloat16
    if device_type == "mps":
        return dtype == torch.float16
    return False


def sync_device(device_type: str) -> None:
    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    raise ValueError("prediction_dtype must be float16 or float32.")


def build_model(base_config: dict) -> HelioSpectFormer:
    return HelioSpectFormer(
        img_size=base_config["model"]["img_size"],
        patch_size=base_config["model"]["patch_size"],
        in_chans=len(base_config["data"]["sdo_channels"]),
        embed_dim=base_config["model"]["embed_dim"],
        time_embedding={
            "type": "linear",
            "time_dim": len(base_config["data"]["time_delta_input_minutes"]),
        },
        depth=base_config["model"]["depth"],
        n_spectral_blocks=base_config["model"]["n_spectral_blocks"],
        num_heads=base_config["model"]["num_heads"],
        mlp_ratio=base_config["model"]["mlp_ratio"],
        drop_rate=base_config["model"]["drop_rate"],
        dtype=torch.bfloat16,
        window_size=base_config["model"]["window_size"],
        dp_rank=base_config["model"]["dp_rank"],
        learned_flow=base_config["model"]["learned_flow"],
        use_latitude_in_learned_flow=base_config["model"]["learned_flow"],
        init_weights=False,
        checkpoint_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        rpe=base_config["model"]["rpe"],
        ensemble=base_config["model"]["ensemble"],
        finetune=base_config["model"]["finetune"],
    )


def required_window_span_minutes(
    time_delta_input_minutes: list[int],
    time_delta_target_minutes: int,
    rollout_steps: int,
) -> int:
    required_offsets = list(time_delta_input_minutes) + [
        (step + 1) * int(time_delta_target_minutes)
        for step in range(int(rollout_steps) + 1)
    ]
    return int(max(required_offsets) - min(required_offsets))


def _datetime_strings(values) -> list[str]:
    return np.asarray(values).astype("datetime64[s]").astype(str).tolist()


def _encode_fixed_width(strings: list[str], width: int) -> np.ndarray:
    arr = np.asarray(strings, dtype=f"S{width}")
    return arr.view("S1").reshape(len(strings), width)


def run_inference_pipeline(
    advanced_cfg: dict[str, Any],
    config_dir: Path,
    prediction_nc_path: Path,
    samples: int,
    show_progress: bool,
) -> InferenceSummary:
    foundation_config_path = _resolve_path(str(advanced_cfg["foundation_config_path"]), config_dir)
    scalers_path = _resolve_path(str(advanced_cfg["scalers_path"]), config_dir)
    weights_path = _resolve_path(str(advanced_cfg["weights_path"]), config_dir)
    index_path = _resolve_path(str(advanced_cfg["index_path"]), config_dir)

    with open(foundation_config_path, "r") as fp:
        base_config = yaml.safe_load(fp)

    base_config["data"]["time_delta_input_minutes"] = [
        int(v) for v in advanced_cfg["time_delta_input_minutes"]
    ]
    base_config["data"]["time_delta_target_minutes"] = int(advanced_cfg["time_delta_target_minutes"])
    base_config["data"]["n_input_timestamps"] = len(base_config["data"]["time_delta_input_minutes"])

    device, device_type = resolve_device(str(advanced_cfg["device"]))
    amp_dtype = resolve_dtype(str(advanced_cfg["dtype"]), device_type)
    if int(advanced_cfg["cpu_threads"]) > 0:
        torch.set_num_threads(int(advanced_cfg["cpu_threads"]))

    scalers_info = yaml.safe_load(open(scalers_path, "r"))
    scalers = build_scalers(info=scalers_info)

    model = build_model(base_config)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    rollout_steps = int(advanced_cfg["rollout_steps"])
    dataset_rollout_steps = int(advanced_cfg["dataset_rollout_steps"])
    dataset = HelioNetCDFDataset(
        index_path=str(index_path),
        time_delta_input_minutes=base_config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=base_config["data"]["time_delta_target_minutes"],
        n_input_timestamps=len(base_config["data"]["time_delta_input_minutes"]),
        rollout_steps=dataset_rollout_steps,
        channels=base_config["data"]["sdo_channels"],
        drop_hmi_probability=base_config["data"]["drop_hmi_probability"],
        num_mask_aia_channels=base_config["data"]["num_mask_aia_channels"],
        use_latitude_in_learned_flow=base_config["data"]["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="valid",
        pooling=base_config["data"]["pooling"],
        random_vert_flip=base_config["data"]["random_vert_flip"],
    )

    if len(dataset) == 0:
        min_span = required_window_span_minutes(
            time_delta_input_minutes=[int(v) for v in base_config["data"]["time_delta_input_minutes"]],
            time_delta_target_minutes=int(base_config["data"]["time_delta_target_minutes"]),
            rollout_steps=dataset_rollout_steps,
        )
        raise RuntimeError(
            "No valid samples in dataset. Increase date range. "
            f"Required coverage is at least {min_span} minutes for rollout={dataset_rollout_steps}."
        )

    dataloader_kwargs = {
        "dataset": dataset,
        "shuffle": False,
        "batch_size": int(advanced_cfg["batch_size"]),
        "num_workers": int(advanced_cfg["num_workers"]),
        "pin_memory": device_type == "cuda",
        "drop_last": False,
        "collate_fn": custom_collate_fn,
    }
    if int(advanced_cfg["num_workers"]) > 0:
        dataloader_kwargs["prefetch_factor"] = int(advanced_cfg["prefetch_factor"])
        dataloader_kwargs["persistent_workers"] = True
    dataloader = DataLoader(**dataloader_kwargs)

    max_batches = min(int(samples), len(dataloader))
    if max_batches <= 0:
        raise RuntimeError("No batches available.")

    channels = list(base_config["data"]["sdo_channels"])
    np_dtype = resolve_numpy_dtype(str(advanced_cfg["prediction_dtype"]))
    means, stds, epsilons, sl_scale_factors = dataset.transformation_inputs()
    img_size = int(base_config["model"]["img_size"])
    if img_size != 4096:
        raise RuntimeError(f"Expected img_size=4096, got {img_size}.")

    def inverse_channel_fn(channel_idx: int, x: np.ndarray) -> np.ndarray:
        return inverse_transform_single_channel(
            x,
            mean=means[channel_idx],
            std=stds[channel_idx],
            epsilon=epsilons[channel_idx],
            sl_scale_factor=sl_scale_factors[channel_idx],
        )

    autocast_enabled = (not bool(advanced_cfg["disable_autocast"])) and supports_autocast(
        device_type=device_type, dtype=amp_dtype
    )
    prediction_steps = rollout_steps + 1
    sample_capacity = max_batches * int(advanced_cfg["batch_size"])
    writer: PredictionNetCDFWriter | None = None

    losses: list[float] = []
    data_times: list[float] = []
    infer_times: list[float] = []
    global_sample_idx = 0

    iterator = iter(dataloader)
    for batch_idx in range(max_batches):
        log_progress(show_progress, f"batch {batch_idx + 1}/{max_batches} | loading data")
        t_data = perf_counter()
        batch_data, batch_metadata = next(iterator)
        data_elapsed = perf_counter() - t_data

        ts = batch_data["ts"].to(device, non_blocking=True)
        time_delta_input = batch_data["time_delta_input"].to(device, non_blocking=True)
        full_h = int(batch_data["ts"].shape[-2])
        full_w = int(batch_data["ts"].shape[-1])
        if full_h != 4096 or full_w != 4096:
            raise RuntimeError(f"Expected input shape 4096x4096, got {full_h}x{full_w}.")

        batch_size = int(batch_data["ts"].shape[0])
        input_steps = int(batch_data["ts"].shape[2])
        gt_steps = int(batch_data["forecast"].shape[2])
        if gt_steps < prediction_steps:
            raise RuntimeError(f"GT steps ({gt_steps}) < prediction steps ({prediction_steps}).")

        if writer is None:
            writer = PredictionNetCDFWriter(
                output_path=str(prediction_nc_path),
                channels=channels,
                prediction_dtype=np_dtype,
                input_steps=input_steps,
                prediction_steps=prediction_steps,
                shape_hw=(full_h, full_w),
                sample_capacity=sample_capacity,
            )
            log_progress(
                show_progress,
                f"initialized prediction.nc writer | shape={sample_capacity}x{prediction_steps}x{full_h}x{full_w}",
            )

        sample_states: list[dict[str, Any]] = []
        for sample_idx_in_batch in range(batch_size):
            sample_global_id = global_sample_idx + sample_idx_in_batch
            timestamps_input = batch_metadata["timestamps_input"][sample_idx_in_batch]
            timestamps_targets = batch_metadata["timestamps_targets"][sample_idx_in_batch]
            writer.write_sample_metadata(
                sample_idx=sample_global_id,
                sample_id=sample_global_id,
                timestamps_input=timestamps_input,
                timestamps_prediction=timestamps_targets[:prediction_steps],
            )
            sample_states.append({"sample_id": sample_global_id})

        t_infer = perf_counter()
        autocast_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype) if autocast_enabled else nullcontext()
        step_losses: list[float] = []
        curr_ts = ts
        with torch.no_grad():
            with autocast_ctx:
                for step in range(prediction_steps):
                    pred = model({"ts": curr_ts, "time_delta_input": time_delta_input})
                    target = batch_data["forecast"][:, :, step, ...].to(device, non_blocking=True)
                    step_losses.append(float(F.mse_loss(pred.float(), target.float()).item()))

                    pred_cpu = pred.detach().cpu().float().numpy()
                    for sample_idx_in_batch, state in enumerate(sample_states):
                        for channel_idx, channel_name in enumerate(channels):
                            pred_frame = pred_cpu[sample_idx_in_batch, channel_idx]
                            pred_inv = inverse_channel_fn(channel_idx, pred_frame.astype(np.float32, copy=False))
                            writer.write_prediction_frame(
                                sample_idx=state["sample_id"],
                                prediction_step_idx=step,
                                channel_name=channel_name,
                                frame_hw=pred_inv.astype(np_dtype, copy=False),
                            )
                    curr_ts = torch.cat((curr_ts[:, :, 1:, ...], pred[:, :, None, ...]), dim=2)

        sync_device(device_type)
        infer_elapsed = perf_counter() - t_infer
        batch_loss = float(np.mean(step_losses))
        log_progress(
            show_progress,
            (
                f"batch {batch_idx + 1}/{max_batches} | done forward | "
                f"loss={batch_loss:.6f} | infer_s={infer_elapsed:.3f}"
            ),
        )

        if batch_idx >= int(advanced_cfg["warmup"]):
            losses.append(batch_loss)
            data_times.append(data_elapsed)
            infer_times.append(infer_elapsed)

        global_sample_idx += batch_size

    if writer is None:
        raise RuntimeError("Prediction writer was not initialized.")
    output_path = writer.finalize(samples_written=global_sample_idx)
    log_progress(show_progress, f"saved prediction.nc | path={output_path}")

    if not infer_times:
        raise RuntimeError("No timed batches after warmup; reduce warmup or increase samples.")
    return InferenceSummary(
        avg_loss=float(np.mean(losses)),
        timed_batches=len(infer_times),
        avg_data_seconds=float(np.mean(data_times)),
        avg_infer_seconds=float(np.mean(infer_times)),
        prediction_nc_path=output_path,
        samples_written=global_sample_idx,
    )


def print_report(
    download_summary: DownloadSummary | None,
    inference_summary: InferenceSummary | None,
    start_dt: datetime,
    end_dt: datetime,
) -> None:
    print("\nEasy Surya Inference")
    print("=" * 72)
    print(f"Window (UTC)         : {_format_datetime(start_dt)} -> {_format_datetime(end_dt)}")
    if download_summary is not None:
        print(
            "Download status      : "
            f"requested={download_summary.requested_timestamps} "
            f"matched={download_summary.matched_timestamps} "
            f"missing={download_summary.missing_timestamps}"
        )
        print(
            "Download files       : "
            f"downloaded={download_summary.downloaded_files} "
            f"skipped={download_summary.skipped_files} "
            f"failed={download_summary.failed_files}"
        )
        print(f"Download output dir  : {download_summary.output_dir}")
    if inference_summary is not None:
        print(f"Samples written      : {inference_summary.samples_written}")
        print(f"Avg rollout MSE      : {inference_summary.avg_loss:.6f}")
        print(f"Avg data sec         : {inference_summary.avg_data_seconds:.3f}")
        print(f"Avg infer sec        : {inference_summary.avg_infer_seconds:.3f}")
        print(f"Prediction file      : {inference_summary.prediction_nc_path}")
    print("=" * 72)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config_path).expanduser().resolve()
    config_dir = config_path.parent

    user_cfg, advanced_cfg = _load_easy_sections(config_path)
    prompt_for_dates = bool(user_cfg["prompt_for_dates"]) and (not args.no_prompt)
    start_dt, end_dt = _select_dates(
        user_cfg=user_cfg,
        cli_start=args.start_datetime,
        cli_end=args.end_datetime,
        use_prompt=prompt_for_dates,
    )

    output_dir = _resolve_path(str(user_cfg["output_dir"]), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_nc_path = output_dir / "prediction.nc"

    validation_data_dir = _resolve_path(str(advanced_cfg["validation_data_dir"]), config_dir)
    index_path = _resolve_path(str(advanced_cfg["index_path"]), config_dir)
    show_progress = bool(advanced_cfg["show_progress"])

    print(f"Easy config          : {config_path}")
    print(
        "Download window      : "
        f"{_format_datetime(start_dt)} -> {_format_datetime(end_dt)} UTC"
    )
    print(f"Validation data dir  : {validation_data_dir}")
    print(f"Index CSV            : {index_path}")
    print(f"Prediction output    : {prediction_nc_path}")
    print(f"Rollout steps        : {int(advanced_cfg['rollout_steps'])}")

    if args.dry_run:
        print("Dry run enabled. No download or inference executed.")
        return 0

    try:
        ensure_model_assets(advanced_cfg=advanced_cfg, config_dir=config_dir)
    except Exception as exc:
        print(f"ERROR downloading model assets: {exc}", file=sys.stderr)
        return 1

    download_summary: DownloadSummary | None = None
    if not args.skip_download:
        try:
            download_summary = download_surya_bench_range(
                bucket=str(advanced_cfg["s3_bucket"]),
                output_dir=validation_data_dir,
                start_datetime=start_dt,
                end_datetime=end_dt,
                cadence_minutes=int(advanced_cfg["cadence_minutes"]),
                skip_existing=bool(advanced_cfg["download_skip_existing"]),
                verify_size=bool(advanced_cfg["download_verify_size"]),
                match_tolerance_minutes=int(advanced_cfg["download_match_tolerance_minutes"]),
                prune_to_expected=bool(advanced_cfg["prune_validation_data_to_window"]),
                show_progress=show_progress,
            )
        except Exception as exc:
            print(f"ERROR during download: {exc}", file=sys.stderr)
            return 1
    else:
        log_progress(show_progress, "skipping download by request")

    build_index_csv_for_range(
        validation_data_dir=validation_data_dir,
        index_path=index_path,
        start_datetime=start_dt,
        end_datetime=end_dt,
        cadence_minutes=int(advanced_cfg["cadence_minutes"]),
    )

    try:
        inference_summary = run_inference_pipeline(
            advanced_cfg=advanced_cfg,
            config_dir=config_dir,
            prediction_nc_path=prediction_nc_path,
            samples=int(user_cfg["samples"]),
            show_progress=show_progress,
        )
    except Exception as exc:
        print(f"ERROR during inference: {exc}", file=sys.stderr)
        return 1

    print_report(
        download_summary=download_summary,
        inference_summary=inference_summary,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

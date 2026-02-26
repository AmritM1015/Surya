#!/usr/bin/env python3
"""Standalone Surya easy inference: prompt dates -> download -> rollout -> prediction.nc."""

from __future__ import annotations

import argparse
import inspect
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import h5netcdf
import numpy as np
import pandas as pd
import skimage.measure
import torch
import yaml
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader, Dataset

from surya.datasets.helio import (
    transform as helio_transform,
)
from surya.models.helio_spectformer import HelioSpectFormer
from surya.utils.data import build_scalers, custom_collate_fn

S3_FILE_PATTERN = re.compile(r"(\d{8})_(\d{4})\.nc$")
DATETIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M")


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
    avg_loss: float | None
    timed_batches: int
    avg_data_seconds: float
    avg_infer_seconds: float
    prediction_nc_path: str
    mode: str


@dataclass
class CoverageSummary:
    total_timestamps: int
    present_timestamps: int
    missing_timestamps: int
    input_complete_references: int
    full_target_references: int
    missing_examples: list[str]


class DebugLogger:
    def __init__(self, enabled: bool, log_path: Path | None) -> None:
        self.enabled = bool(enabled)
        self.log_path = str(log_path.resolve()) if (self.enabled and log_path is not None) else ""
        self._fp = None
        self._line_number = 0
        if self.enabled:
            if log_path is None:
                raise ValueError("debug_mode is enabled but debug_log_path is not set.")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = open(log_path, "w", encoding="utf-8", buffering=1)
            self.log("debug_started", log_path=self.log_path)

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            value_float = float(value)
            if np.isfinite(value_float):
                return f"{value_float:.6f}"
            return str(value_float)
        if isinstance(value, (list, tuple)):
            return "[" + ", ".join(str(item) for item in value) + "]"
        if isinstance(value, dict):
            entries = ", ".join(f"{key}={value[key]}" for key in sorted(value))
            return "{" + entries + "}"
        return str(value)

    @staticmethod
    def _caller_location() -> str:
        frame = inspect.currentframe()
        try:
            if frame is None:
                return "unknown:0"
            log_frame = frame.f_back
            if log_frame is None:
                return "unknown:0"
            caller_frame = log_frame.f_back
            if caller_frame is None:
                return "unknown:0"
            caller_file = Path(caller_frame.f_code.co_filename).name
            return f"{caller_file}:{caller_frame.f_lineno}"
        finally:
            del frame

    def log(self, event: str, **payload: Any) -> None:
        if not self.enabled or self._fp is None:
            return
        self._line_number += 1
        ts_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        location = self._caller_location()
        payload_items = [f"{key}={self._format_value(value)}" for key, value in sorted(payload.items())]
        payload_text = " | ".join(payload_items)
        line = f"{self._line_number:06d} | {ts_utc} | {location} | {event}"
        if payload_text:
            line = f"{line} | {payload_text}"
        self._fp.write(line + "\n")

    def close(self) -> None:
        if self._fp is not None:
            self.log("debug_finished")
            self._fp.close()
            self._fp = None


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
        self.file.attrs["data_layout"] = (
            "prediction vars: <channel>, ground truth vars: gt_<channel>, "
            "all dims=(sample,prediction_time,y,x)"
        )
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
        self.ground_truth_vars: dict[str, Any] = {}
        for channel in self.channels:
            self.prediction_vars[channel] = self.file.create_variable(
                channel,
                ("sample", "prediction_time", "y", "x"),
                dtype=self.prediction_dtype,
            )
            gt_name = f"gt_{channel}"
            self.ground_truth_vars[gt_name] = self.file.create_variable(
                gt_name,
                ("sample", "prediction_time", "y", "x"),
                dtype=self.prediction_dtype,
            )

        self.file.attrs["prediction_variables"] = ",".join(self.channels)
        self.file.attrs["ground_truth_variables"] = ",".join(
            [f"gt_{channel}" for channel in self.channels]
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

    def write_ground_truth_frame(
        self,
        sample_idx: int,
        prediction_step_idx: int,
        channel_name: str,
        frame_hw: np.ndarray,
    ) -> None:
        gt_name = f"gt_{channel_name}"
        self.ground_truth_vars[gt_name][sample_idx, prediction_step_idx, :, :] = frame_hw

    def finalize(self, samples_written: int) -> str:
        self.file.attrs["samples_written"] = int(samples_written)
        self.file.close()
        return self.output_path


class InputOnlyRolloutDataset(Dataset):
    """Input-complete dataset that tolerates missing targets by masking them."""

    def __init__(
        self,
        present_index: pd.DataFrame,
        reference_timestamps: list[pd.Timestamp],
        channels: list[str],
        time_delta_input_minutes: list[int],
        time_delta_target_minutes: int,
        prediction_steps: int,
        scalers: dict[str, Any],
        pooling: int,
        debug_logger: DebugLogger | None = None,
    ) -> None:
        self.present_index = present_index.copy()
        self.reference_timestamps = list(reference_timestamps)
        self.channels = list(channels)
        self.time_delta_input_minutes = sorted(int(v) for v in time_delta_input_minutes)
        self.time_delta_target_minutes = int(time_delta_target_minutes)
        self.prediction_steps = int(prediction_steps)
        self.scalers = scalers
        self.pooling = int(pooling) if pooling is not None else 1
        self.debug_logger = debug_logger

        if len(self.reference_timestamps) == 0:
            raise ValueError("InputOnlyRolloutDataset requires at least one reference timestamp.")

        self.present_index["path"] = self.present_index["path"].astype(str)
        self.path_lookup = self.present_index["path"].to_dict()
        self.latest_input_offset_minutes = self.time_delta_input_minutes[-1]

    def __len__(self) -> int:
        return len(self.reference_timestamps)

    @cache
    def transformation_inputs(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        means = np.array([self.scalers[ch].mean for ch in self.channels])
        stds = np.array([self.scalers[ch].std for ch in self.channels])
        epsilons = np.array([self.scalers[ch].epsilon for ch in self.channels])
        sl_scale_factors = np.array([self.scalers[ch].sl_scale_factor for ch in self.channels])
        return means, stds, epsilons, sl_scale_factors

    def _load_transformed_frame(self, timestep: pd.Timestamp) -> np.ndarray:
        if timestep not in self.path_lookup:
            raise KeyError(f"Timestamp not present in index: {timestep}")
        filepath = Path(self.path_lookup[timestep]).expanduser()
        if not filepath.is_absolute():
            filepath = (Path.cwd() / filepath).resolve()
        t_read = perf_counter()
        data = _read_channels_frame(filepath=filepath, channels=self.channels)
        read_s = perf_counter() - t_read

        t_pool = perf_counter()
        if self.pooling > 1:
            data = skimage.measure.block_reduce(
                data, block_size=(1, self.pooling, self.pooling), func=np.mean
            )
        pool_s = perf_counter() - t_pool

        t_transform = perf_counter()
        means, stds, epsilons, sl_scale_factors = self.transformation_inputs()
        transformed = helio_transform(data, means, stds, sl_scale_factors, epsilons)
        transformed = transformed.astype(np.float32, copy=False)
        transform_s = perf_counter() - t_transform
        if self.debug_logger is not None and self.debug_logger.enabled:
            self.debug_logger.log(
                "input_frame_loaded",
                timestep=str(pd.Timestamp(timestep)),
                filepath=str(filepath),
                read_s=read_s,
                pool_s=pool_s,
                transform_s=transform_s,
                shape_chw=list(transformed.shape),
            )
        return transformed

    def __getitem__(self, idx: int) -> tuple[dict[str, Any], dict[str, Any]]:
        reference_timestep = self.reference_timestamps[idx]
        input_timestamps = [
            reference_timestep + timedelta(minutes=offset) for offset in self.time_delta_input_minutes
        ]
        input_frames = [self._load_transformed_frame(ts) for ts in input_timestamps]
        stacked_inputs = np.stack(input_frames, axis=1).astype(np.float32, copy=False)

        target_timestamps: list[np.datetime64] = []
        for step in range(self.prediction_steps):
            ts = reference_timestep + timedelta(
                minutes=(step + 1) * self.time_delta_target_minutes
            )
            target_timestamps.append(np.datetime64(ts))

        time_delta_input_float = (
            self.latest_input_offset_minutes
            - np.asarray(self.time_delta_input_minutes, dtype=np.float32)
        ) / 60.0
        lead_time_delta_float = (
            self.latest_input_offset_minutes
            - np.asarray(
                [(step + 1) * self.time_delta_target_minutes for step in range(self.prediction_steps)],
                dtype=np.float32,
            )
        ) / 60.0

        batch_data = {
            "ts": stacked_inputs,
            "time_delta_input": time_delta_input_float.astype(np.float32, copy=False),
            "lead_time_delta": lead_time_delta_float.astype(np.float32, copy=False),
        }
        metadata = {
            "timestamps_input": np.asarray(input_timestamps, dtype="datetime64[ns]"),
            "timestamps_targets": np.asarray(target_timestamps, dtype="datetime64[ns]"),
        }
        return batch_data, metadata


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
        "--rollout-steps",
        type=int,
        default=None,
        help="Override user.rollout_steps.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Disable interactive date/rollout prompt.",
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


def _parse_rollout_steps(value: Any) -> int:
    rollout_steps = int(value)
    if rollout_steps < 0:
        raise ValueError("rollout_steps must be >= 0.")
    return rollout_steps


def _prompt_rollout_steps(default_value: int) -> int:
    default_text = str(int(default_value))
    while True:
        value = input(f"Rollout steps [{default_text}]: ").strip()
        if value == "":
            return int(default_value)
        try:
            return _parse_rollout_steps(value)
        except (TypeError, ValueError) as exc:
            print(f"Invalid rollout_steps '{value}': {exc}")


def _load_easy_sections(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(config_path, "r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp)

    if raw is None:
        raise ValueError(f"Easy config is empty: {config_path}")
    if not isinstance(raw, dict):
        raise ValueError(f"Easy config must be a mapping. Got {type(raw).__name__}.")
    if "user" not in raw or "advanced" not in raw:
        raise ValueError("Easy config must contain top-level 'user' and 'advanced' sections.")

    raw_user = raw["user"]
    raw_advanced = raw["advanced"]
    if not isinstance(raw_user, dict):
        raise ValueError(f"'user' section must be a mapping, got {type(raw_user).__name__}.")
    if not isinstance(raw_advanced, dict):
        raise ValueError(f"'advanced' section must be a mapping, got {type(raw_advanced).__name__}.")
    return dict(raw_user), dict(raw_advanced)


def _create_debug_logger(
    advanced_cfg: dict[str, Any],
    output_dir: Path,
    config_dir: Path,
) -> DebugLogger:
    debug_enabled = bool(advanced_cfg["debug_mode"])
    debug_log_path_raw = str(advanced_cfg["debug_log_path"]).strip()
    if not debug_enabled:
        return DebugLogger(enabled=False, log_path=None)

    if debug_log_path_raw:
        debug_log_path = _resolve_path(debug_log_path_raw, config_dir)
    else:
        debug_log_path = (output_dir / "inference_debug.txt").resolve()
    return DebugLogger(enabled=True, log_path=debug_log_path)


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


def _select_rollout_steps(
    user_cfg: dict[str, Any],
    cli_rollout_steps: int | None,
    use_prompt: bool,
) -> int:
    if cli_rollout_steps is not None:
        rollout_steps = _parse_rollout_steps(cli_rollout_steps)
    else:
        rollout_steps = _parse_rollout_steps(user_cfg["rollout_steps"])
    if use_prompt:
        rollout_steps = _prompt_rollout_steps(rollout_steps)
    return rollout_steps


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
    if missing_timestamps:
        log_progress(show_progress, "download status | [v]=available [x]=missing")
        for ts in missing_timestamps:
            log_progress(show_progress, f"download status | [x] {_format_datetime(ts)}")
    else:
        log_progress(show_progress, "download status | [v] all expected timestamps available")

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
    if device_type == "cpu":
        cpu_bf16_supported = hasattr(torch.cpu, "is_bf16_supported") and bool(
            torch.cpu.is_bf16_supported()
        )
        return torch.bfloat16 if cpu_bf16_supported else torch.float32
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


def _reset_peak_memory_stats(device_type: str, device: torch.device) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device=device)


def _memory_stats_mb(device_type: str, device: torch.device) -> dict[str, float]:
    if device_type == "cuda" and torch.cuda.is_available():
        return {
            "mem_allocated_mb": float(torch.cuda.memory_allocated(device=device) / (1024**2)),
            "mem_reserved_mb": float(torch.cuda.memory_reserved(device=device) / (1024**2)),
            "mem_peak_allocated_mb": float(torch.cuda.max_memory_allocated(device=device) / (1024**2)),
            "mem_peak_reserved_mb": float(torch.cuda.max_memory_reserved(device=device) / (1024**2)),
        }
    if device_type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return {"mem_allocated_mb": float(torch.mps.current_allocated_memory() / (1024**2))}
    return {}


def _read_channels_frame(filepath: Path, channels: list[str]) -> np.ndarray:
    with h5netcdf.File(filepath, "r") as nc:
        return np.stack([np.asarray(nc.variables[ch][:]) for ch in channels], axis=0)


def resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    if dtype_name == "float16":
        return np.float16
    if dtype_name == "float32":
        return np.float32
    raise ValueError("prediction_dtype must be float16 or float32.")


def build_model(base_config: dict) -> HelioSpectFormer:
    model_depth = int(base_config["model"]["depth"])
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
        checkpoint_layers=list(range(model_depth)),
        rpe=base_config["model"]["rpe"],
        ensemble=base_config["model"]["ensemble"],
        finetune=base_config["model"]["finetune"],
    )


def _index_coverage_summary(
    index_path: Path,
    time_delta_input_minutes: list[int],
    time_delta_target_minutes: int,
    full_target_steps: int,
) -> tuple[CoverageSummary, pd.DataFrame, list[pd.Timestamp], list[pd.Timestamp]]:
    required_columns = {"path", "timestep", "present"}
    index_df = pd.read_csv(index_path)
    missing_columns = required_columns.difference(index_df.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise RuntimeError(f"Index CSV is missing required columns: {missing_list}")

    index_df["timestep"] = pd.to_datetime(index_df["timestep"])
    index_df.sort_values("timestep", inplace=True)
    present_df = index_df[index_df["present"] == 1].copy()
    present_df.set_index("timestep", inplace=True)
    present_df.sort_index(inplace=True)
    present_set = set(present_df.index)

    input_offsets = [timedelta(minutes=int(v)) for v in sorted(time_delta_input_minutes)]
    target_offsets = [
        timedelta(minutes=(step + 1) * int(time_delta_target_minutes))
        for step in range(int(full_target_steps))
    ]

    input_complete_refs: list[pd.Timestamp] = []
    full_target_refs: list[pd.Timestamp] = []
    for reference_timestep in sorted(present_set):
        if all((reference_timestep + offset) in present_set for offset in input_offsets):
            input_complete_refs.append(reference_timestep)
            if all((reference_timestep + offset) in present_set for offset in target_offsets):
                full_target_refs.append(reference_timestep)

    missing_examples = (
        index_df[index_df["present"] != 1]["timestep"]
        .dt.strftime("%Y-%m-%d %H:%M:%S")
        .head(8)
        .tolist()
    )
    coverage = CoverageSummary(
        total_timestamps=len(index_df),
        present_timestamps=len(present_df),
        missing_timestamps=int((index_df["present"] != 1).sum()),
        input_complete_references=len(input_complete_refs),
        full_target_references=len(full_target_refs),
        missing_examples=missing_examples,
    )
    return coverage, present_df, input_complete_refs, full_target_refs


def _synthesize_prediction_timestamps(
    timestamps_input,
    target_delta_minutes: int,
    prediction_steps: int,
) -> np.ndarray:
    input_arr = np.asarray(timestamps_input).astype("datetime64[ns]")
    if input_arr.size == 0:
        raise ValueError("Cannot synthesize prediction timestamps without input timestamps.")
    last_input = pd.Timestamp(input_arr[-1]).to_pydatetime()
    synthesized = [
        np.datetime64(last_input + timedelta(minutes=(step + 1) * int(target_delta_minutes)))
        for step in range(int(prediction_steps))
    ]
    return np.asarray(synthesized, dtype="datetime64[ns]")


def _datetime_strings(values) -> list[str]:
    return np.asarray(values).astype("datetime64[s]").astype(str).tolist()


def _format_items_for_log(values: list[str], max_items: int | None = None) -> str:
    if max_items is not None and len(values) > max_items:
        head = values[:max_items]
        remaining = len(values) - max_items
        return "[" + ", ".join(head) + f", ... (+{remaining} more)]"
    return "[" + ", ".join(values) + "]"


def _encode_fixed_width(strings: list[str], width: int) -> np.ndarray:
    arr = np.asarray(strings, dtype=f"S{width}")
    return arr.view("S1").reshape(len(strings), width)


def _validate_rollout_against_window(
    start_dt: datetime,
    end_dt: datetime,
    time_delta_input_minutes: list[int],
    time_delta_target_minutes: int,
    rollout_steps: int,
) -> None:
    input_offsets = [int(v) for v in time_delta_input_minutes]
    earliest_reference = pd.Timestamp(start_dt) - pd.Timedelta(minutes=min(input_offsets))
    target_delta_minutes = int(time_delta_target_minutes)
    max_steps_fit = int(
        (pd.Timestamp(end_dt) - earliest_reference).total_seconds() // (target_delta_minutes * 60)
    )
    max_rollout = max_steps_fit - 1
    if rollout_steps <= max_rollout:
        return

    required_end = earliest_reference.to_pydatetime() + timedelta(
        minutes=(int(rollout_steps) + 1) * target_delta_minutes
    )
    suggested_rollout = max(0, max_rollout)
    raise ValueError(
        "Requested rollout_steps does not fit inside the selected window.\n"
        f"  start_datetime: {_format_datetime(start_dt)} UTC\n"
        f"  end_datetime  : {_format_datetime(end_dt)} UTC\n"
        f"  first reference timestep: {_format_datetime(earliest_reference.to_pydatetime())} UTC\n"
        f"  requested rollout_steps : {int(rollout_steps)}\n"
        f"  max rollout_steps allowed for this window: {suggested_rollout}\n"
        f"Suggestion:\n"
        f"  1) Set user.rollout_steps to <= {suggested_rollout}\n"
        f"  2) Or extend user.end_datetime to >= {_format_datetime(required_end)} UTC"
    )


def _log_rollout_plan(
    show_progress: bool,
    time_delta_input_minutes: list[int],
    target_delta_minutes: int,
    prediction_steps: int,
) -> None:
    input_offsets_minutes = sorted(int(v) for v in time_delta_input_minutes)
    log_progress(
        show_progress,
        (
            "rollout plan | "
            f"input_offsets_min={input_offsets_minutes} "
            f"target_delta_min={int(target_delta_minutes)} "
            f"prediction_steps={int(prediction_steps)}"
        ),
    )


def _raise_input_missing_error(
    coverage: CoverageSummary,
    time_delta_input_minutes: list[int],
) -> None:
    input_deltas = [int(v) for v in time_delta_input_minutes]
    input_span_minutes = int(max(input_deltas) - min(input_deltas))
    missing_preview = ", ".join(coverage.missing_examples) if coverage.missing_examples else "none"
    raise RuntimeError(
        "Input data is missing for required input offsets. "
        f"Required input offsets={input_deltas} (span={input_span_minutes} minutes). "
        f"Index present={coverage.present_timestamps}, missing={coverage.missing_timestamps}. "
        f"Missing timestamp sample: {missing_preview}."
    )


def _build_single_sample_dataloader(
    dataset: Dataset,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> DataLoader:
    num_workers = int(num_workers)
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "shuffle": False,
        "batch_size": 1,
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "drop_last": False,
        "collate_fn": custom_collate_fn,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
        kwargs["persistent_workers"] = True
    return DataLoader(**kwargs)


def _build_inverse_batch_fn(dataset: InputOnlyRolloutDataset) -> Callable[[np.ndarray], np.ndarray]:
    means, stds, epsilons, sl_scale_factors = dataset.transformation_inputs()
    means_hw = means[:, None, None].astype(np.float32, copy=False)
    stds_hw = stds[:, None, None].astype(np.float32, copy=False)
    eps_hw = epsilons[:, None, None].astype(np.float32, copy=False)
    scales_hw = sl_scale_factors[:, None, None].astype(np.float32, copy=False)

    def inverse_batch_fn(prediction_chw: np.ndarray) -> np.ndarray:
        restored = prediction_chw.astype(np.float32, copy=False) * (stds_hw + eps_hw) + means_hw
        restored = np.sign(restored) * np.expm1(np.abs(restored))
        restored = restored / scales_hw
        return restored.astype(np.float32, copy=False)

    return inverse_batch_fn


def _build_gt_frame_loader(
    present_index: pd.DataFrame,
    channels: list[str],
    pooling: int,
    prediction_dtype: np.dtype,
    debug_logger: DebugLogger | None = None,
) -> Callable[[Any], np.ndarray | None]:
    path_lookup = present_index["path"].to_dict()

    def load_gt_frame_for_prediction_timestamp(ts_value: Any) -> np.ndarray | None:
        t_total = perf_counter()
        ts = pd.Timestamp(np.asarray(ts_value).astype("datetime64[ns]").item())
        path_string = path_lookup.get(ts)
        if path_string is None:
            if debug_logger is not None and debug_logger.enabled:
                debug_logger.log(
                    "gt_frame_loaded",
                    timestep=str(ts),
                    found=False,
                    total_s=perf_counter() - t_total,
                )
            return None
        filepath = Path(path_string).expanduser()
        if not filepath.is_absolute():
            filepath = (Path.cwd() / filepath).resolve()
        t_read = perf_counter()
        frame = _read_channels_frame(filepath=filepath, channels=channels)
        read_s = perf_counter() - t_read
        t_pool = perf_counter()
        if pooling > 1:
            frame = skimage.measure.block_reduce(frame, block_size=(1, pooling, pooling), func=np.mean)
        pool_s = perf_counter() - t_pool
        frame = frame.astype(prediction_dtype, copy=False)
        if debug_logger is not None and debug_logger.enabled:
            debug_logger.log(
                "gt_frame_loaded",
                timestep=str(ts),
                found=True,
                filepath=str(filepath),
                read_s=read_s,
                pool_s=pool_s,
                total_s=perf_counter() - t_total,
                shape_chw=list(frame.shape),
            )
        return frame

    return load_gt_frame_for_prediction_timestamp


def _prepare_inference_context(
    advanced_cfg: dict[str, Any],
    config_dir: Path,
    rollout_steps: int,
    debug_logger: DebugLogger | None = None,
) -> dict[str, Any]:
    foundation_config_path = _resolve_path(str(advanced_cfg["foundation_config_path"]), config_dir)
    scalers_path = _resolve_path(str(advanced_cfg["scalers_path"]), config_dir)
    weights_path = _resolve_path(str(advanced_cfg["weights_path"]), config_dir)
    index_path = _resolve_path(str(advanced_cfg["index_path"]), config_dir)

    with open(foundation_config_path, "r") as fp:
        base_config = yaml.safe_load(fp)
    base_config["data"]["time_delta_input_minutes"] = [int(v) for v in advanced_cfg["time_delta_input_minutes"]]
    base_config["data"]["time_delta_target_minutes"] = int(advanced_cfg["time_delta_target_minutes"])
    base_config["data"]["n_input_timestamps"] = len(base_config["data"]["time_delta_input_minutes"])

    device, device_type = resolve_device(str(advanced_cfg["device"]))
    amp_dtype = resolve_dtype(str(advanced_cfg["dtype"]), device_type)
    if device_type == "cuda":
        tf32_enabled = bool(advanced_cfg["enable_tf32"])
        torch.backends.cuda.matmul.allow_tf32 = tf32_enabled
        torch.backends.cudnn.allow_tf32 = tf32_enabled
        torch.backends.cudnn.benchmark = bool(advanced_cfg["enable_cudnn_benchmark"])
    if int(advanced_cfg["cpu_threads"]) > 0:
        torch.set_num_threads(int(advanced_cfg["cpu_threads"]))

    with open(scalers_path, "r") as fp:
        scalers = build_scalers(info=yaml.safe_load(fp))

    model = build_model(base_config)
    weights = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()

    prediction_steps = int(rollout_steps) + 1
    autocast_enabled = (not bool(advanced_cfg["disable_autocast"])) and supports_autocast(
        device_type=device_type, dtype=amp_dtype
    )
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "prepare_context_done",
            device=str(device),
            device_type=device_type,
            amp_dtype=str(amp_dtype),
            autocast_enabled=autocast_enabled,
            prediction_steps=prediction_steps,
        )
    return {
        "base_config": base_config,
        "scalers": scalers,
        "model": model,
        "device": device,
        "device_type": device_type,
        "amp_dtype": amp_dtype,
        "autocast_enabled": autocast_enabled,
        "index_path": index_path,
        "prediction_steps": prediction_steps,
    }


def _prepare_inference_data(
    advanced_cfg: dict[str, Any],
    context: dict[str, Any],
    show_progress: bool,
    debug_logger: DebugLogger | None = None,
) -> dict[str, Any]:
    base_config = context["base_config"]
    prediction_steps = int(context["prediction_steps"])
    time_delta_input_minutes = [int(v) for v in base_config["data"]["time_delta_input_minutes"]]
    target_delta_minutes = int(base_config["data"]["time_delta_target_minutes"])

    coverage, present_index, input_complete_refs, _ = _index_coverage_summary(
        index_path=context["index_path"],
        time_delta_input_minutes=time_delta_input_minutes,
        time_delta_target_minutes=target_delta_minutes,
        full_target_steps=prediction_steps,
    )
    _log_rollout_plan(
        show_progress=show_progress,
        time_delta_input_minutes=time_delta_input_minutes,
        target_delta_minutes=target_delta_minutes,
        prediction_steps=prediction_steps,
    )
    if coverage.input_complete_references == 0:
        _raise_input_missing_error(coverage=coverage, time_delta_input_minutes=time_delta_input_minutes)
    if coverage.full_target_references == 0:
        print(
            "[warning] Missing target coverage detected. Inference will continue with available GT only.",
            flush=True,
        )

    dataset = InputOnlyRolloutDataset(
        present_index=present_index,
        reference_timestamps=input_complete_refs,
        channels=list(base_config["data"]["sdo_channels"]),
        time_delta_input_minutes=time_delta_input_minutes,
        time_delta_target_minutes=target_delta_minutes,
        prediction_steps=prediction_steps,
        scalers=context["scalers"],
        pooling=int(base_config["data"]["pooling"]),
        debug_logger=debug_logger,
    )

    dataloader = _build_single_sample_dataloader(
        dataset=dataset,
        num_workers=int(advanced_cfg["num_workers"]),
        prefetch_factor=int(advanced_cfg["prefetch_factor"]),
        pin_memory=context["device_type"] == "cuda",
    )
    if len(dataloader) <= 0:
        raise RuntimeError("No batches available after filtering for required input data.")

    channels = list(base_config["data"]["sdo_channels"])
    pooling = int(base_config["data"]["pooling"])
    np_dtype = resolve_numpy_dtype(str(advanced_cfg["prediction_dtype"]))
    expected_img_size = int(base_config["model"]["img_size"])
    if expected_img_size <= 0:
        raise RuntimeError(f"Expected positive img_size, got {expected_img_size}.")
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "prepare_data_done",
            index_total=coverage.total_timestamps,
            index_present=coverage.present_timestamps,
            index_missing=coverage.missing_timestamps,
            input_complete_refs=coverage.input_complete_references,
            full_target_refs=coverage.full_target_references,
            dataset_len=len(dataset),
            dataloader_len=len(dataloader),
            channels=len(channels),
            expected_img_size=expected_img_size,
        )

    return {
        "dataloader": dataloader,
        "channels": channels,
        "np_dtype": np_dtype,
        "prediction_steps": prediction_steps,
        "target_delta_minutes": target_delta_minutes,
        "expected_img_size": expected_img_size,
        "inverse_batch_fn": _build_inverse_batch_fn(dataset),
        "load_gt_frame_for_prediction_timestamp": _build_gt_frame_loader(
            present_index=present_index,
            channels=channels,
            pooling=pooling,
            prediction_dtype=np_dtype,
            debug_logger=debug_logger,
        ),
    }


def _validate_single_sample_tensor_shape(
    ts_tensor: torch.Tensor,
    expected_img_size: int,
) -> tuple[int, int]:
    full_h, full_w = int(ts_tensor.shape[-2]), int(ts_tensor.shape[-1])
    if full_h != expected_img_size or full_w != expected_img_size:
        raise RuntimeError(
            f"Expected input shape {expected_img_size}x{expected_img_size}, got {full_h}x{full_w}."
        )
    if int(ts_tensor.shape[0]) != 1:
        raise RuntimeError(f"Easy inference expects batch size 1, got {int(ts_tensor.shape[0])}.")
    return full_h, full_w


def _init_single_sample_io(
    batch_data: dict[str, Any],
    batch_metadata: dict[str, Any],
    prediction_nc_path: Path,
    channels: list[str],
    prediction_steps: int,
    prediction_dtype: np.dtype,
    target_delta_minutes: int,
    expected_img_size: int,
    show_progress: bool,
    debug_logger: DebugLogger | None = None,
) -> dict[str, Any]:
    ts_tensor = batch_data["ts"]
    full_h, full_w = _validate_single_sample_tensor_shape(
        ts_tensor=ts_tensor,
        expected_img_size=expected_img_size,
    )

    writer = PredictionNetCDFWriter(
        output_path=str(prediction_nc_path),
        channels=channels,
        prediction_dtype=prediction_dtype,
        input_steps=int(ts_tensor.shape[2]),
        prediction_steps=prediction_steps,
        shape_hw=(full_h, full_w),
        sample_capacity=1,
    )
    log_progress(show_progress, f"initialized prediction.nc writer | shape=1x{prediction_steps}x{full_h}x{full_w}")
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "writer_initialized",
            prediction_steps=prediction_steps,
            shape_hw=[full_h, full_w],
            output_path=str(prediction_nc_path),
        )

    sample_id = 0
    input_timestamps = np.asarray(batch_metadata["timestamps_input"][0]).astype("datetime64[ns]")
    prediction_timestamps = _synthesize_prediction_timestamps(
        timestamps_input=input_timestamps,
        target_delta_minutes=target_delta_minutes,
        prediction_steps=prediction_steps,
    )
    writer.write_sample_metadata(
        sample_idx=sample_id,
        sample_id=sample_id,
        timestamps_input=input_timestamps,
        timestamps_prediction=prediction_timestamps,
    )
    input_labels = [f"GT_{ts}" for ts in _datetime_strings(input_timestamps)]
    output_labels = [
        f"PR_{step + 1}_{ts}" for step, ts in enumerate(_datetime_strings(prediction_timestamps))
    ]
    return {
        "writer": writer,
        "sample_id": sample_id,
        "input_labels": input_labels,
        "output_labels": output_labels,
        "prediction_timestamps": prediction_timestamps,
        "nan_frame_hw": np.full((full_h, full_w), np.nan, dtype=prediction_dtype),
    }


def _write_step_predictions_and_gt(
    writer: PredictionNetCDFWriter,
    sample_id: int,
    step: int,
    channels: list[str],
    prediction: np.ndarray,
    gt_frame: np.ndarray | None,
    inverse_batch_fn: Callable[[np.ndarray], np.ndarray],
    prediction_dtype: np.dtype,
    nan_frame_hw: np.ndarray,
) -> tuple[float | None, float, float]:
    t_inverse = perf_counter()
    pred_inv_chw = inverse_batch_fn(prediction)
    inverse_s = perf_counter() - t_inverse
    step_loss = None
    if gt_frame is not None:
        diff = pred_inv_chw.astype(np.float32, copy=False) - gt_frame.astype(np.float32, copy=False)
        step_loss = float(np.mean(diff * diff))

    t_write = perf_counter()
    for channel_idx, channel_name in enumerate(channels):
        pred_inv = pred_inv_chw[channel_idx]
        writer.write_prediction_frame(
            sample_idx=sample_id,
            prediction_step_idx=step,
            channel_name=channel_name,
            frame_hw=pred_inv.astype(prediction_dtype, copy=False),
        )
        gt_hw = nan_frame_hw if gt_frame is None else gt_frame[channel_idx]
        writer.write_ground_truth_frame(
            sample_idx=sample_id,
            prediction_step_idx=step,
            channel_name=channel_name,
            frame_hw=gt_hw,
        )
    write_s = perf_counter() - t_write
    return step_loss, inverse_s, write_s


def _run_single_sample_rollout(
    model,
    ts: torch.Tensor,
    time_delta_input: torch.Tensor,
    writer: PredictionNetCDFWriter,
    sample_id: int,
    channels: list[str],
    prediction_steps: int,
    input_labels: list[str],
    output_labels: list[str],
    prediction_timestamps: np.ndarray,
    inverse_batch_fn,
    load_gt_frame_for_prediction_timestamp,
    prediction_dtype: np.dtype,
    nan_frame_hw: np.ndarray,
    device: torch.device,
    device_type: str,
    amp_dtype: torch.dtype,
    autocast_enabled: bool,
    gt_prefetch_workers: int,
    show_progress: bool,
    debug_logger: DebugLogger | None = None,
) -> tuple[list[float], int, int, float]:
    t_infer = perf_counter()
    step_losses: list[float] = []
    gt_pairs_used = 0
    gt_pairs_expected = 0
    curr_ts = ts
    window_labels = list(input_labels)
    autocast_ctx = torch.autocast(device_type=device_type, dtype=amp_dtype) if autocast_enabled else nullcontext()

    prefetch_workers = max(1, int(gt_prefetch_workers))
    _reset_peak_memory_stats(device_type=device_type, device=device)
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "rollout_started",
            prediction_steps=prediction_steps,
            prefetch_workers=prefetch_workers,
            device_type=device_type,
            amp_dtype=str(amp_dtype),
            autocast_enabled=autocast_enabled,
        )

    with ThreadPoolExecutor(max_workers=prefetch_workers) as io_executor:
        next_gt_future: Future[np.ndarray | None] | None = None
        if prediction_steps > 0:
            next_gt_future = io_executor.submit(
                load_gt_frame_for_prediction_timestamp, prediction_timestamps[0]
            )

        with torch.inference_mode():
            with autocast_ctx:
                for step in range(prediction_steps):
                    t_step = perf_counter()
                    t_forward = perf_counter()
                    pred = model({"ts": curr_ts, "time_delta_input": time_delta_input})
                    forward_s = perf_counter() - t_forward
                    gt_pairs_expected += 1
                    if step == 0:
                        log_progress(
                            show_progress,
                            f"batch 1/1 | model_shapes in={tuple(curr_ts.shape)} out={tuple(pred.shape)}",
                        )

                    if step + 1 < prediction_steps:
                        next_ts = prediction_timestamps[step + 1]
                        queued_future = io_executor.submit(
                            load_gt_frame_for_prediction_timestamp, next_ts
                        )
                    else:
                        queued_future = None

                    t_to_cpu = perf_counter()
                    pred_cpu = pred.detach().cpu().float().numpy()[0]
                    to_cpu_s = perf_counter() - t_to_cpu
                    t_wait_gt = perf_counter()
                    gt_frame = next_gt_future.result() if next_gt_future is not None else None
                    wait_gt_s = perf_counter() - t_wait_gt
                    next_gt_future = queued_future
                    if gt_frame is not None:
                        gt_pairs_used += 1

                    step_loss, inverse_s, write_s = _write_step_predictions_and_gt(
                        writer=writer,
                        sample_id=sample_id,
                        step=step,
                        channels=channels,
                        prediction=pred_cpu,
                        gt_frame=gt_frame,
                        inverse_batch_fn=inverse_batch_fn,
                        prediction_dtype=prediction_dtype,
                        nan_frame_hw=nan_frame_hw,
                    )
                    if step_loss is not None:
                        step_losses.append(step_loss)
                    loss_text = f"loss={step_loss:.6f}" if step_loss is not None else "GT missing"
                    output_label = output_labels[step]
                    step_input_labels = list(window_labels)
                    log_progress(
                        show_progress,
                        f"infer step {step + 1}/{prediction_steps} | "
                        f"in={_format_items_for_log(window_labels, max_items=32)} -> out={output_label} | {loss_text}",
                    )
                    window_labels = window_labels[1:] + [output_label]
                    curr_ts = torch.cat((curr_ts[:, :, 1:, ...], pred[:, :, None, ...]), dim=2)
                    if debug_logger is not None and debug_logger.enabled:
                        debug_logger.log(
                            "infer_step",
                            step=step + 1,
                            steps_total=prediction_steps,
                            input_labels=step_input_labels,
                            output_label=output_label,
                            gt_available=gt_frame is not None,
                            loss=step_loss,
                            forward_s=forward_s,
                            pred_to_cpu_s=to_cpu_s,
                            gt_wait_s=wait_gt_s,
                            inverse_transform_s=inverse_s,
                            write_s=write_s,
                            step_total_s=perf_counter() - t_step,
                            **_memory_stats_mb(device_type=device_type, device=device),
                        )

    sync_device(device_type)
    return step_losses, gt_pairs_used, gt_pairs_expected, (perf_counter() - t_infer)


def run_inference_pipeline(
    advanced_cfg: dict[str, Any],
    config_dir: Path,
    prediction_nc_path: Path,
    rollout_steps: int,
    show_progress: bool,
    debug_logger: DebugLogger | None = None,
) -> InferenceSummary:
    t_context = perf_counter()
    context = _prepare_inference_context(
        advanced_cfg=advanced_cfg,
        config_dir=config_dir,
        rollout_steps=rollout_steps,
        debug_logger=debug_logger,
    )
    context_s = perf_counter() - t_context

    t_prepare_data = perf_counter()
    data = _prepare_inference_data(
        advanced_cfg=advanced_cfg,
        context=context,
        show_progress=show_progress,
        debug_logger=debug_logger,
    )
    prepare_data_s = perf_counter() - t_prepare_data
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "inference_prepare_complete",
            context_s=context_s,
            prepare_data_s=prepare_data_s,
        )

    iterator = iter(data["dataloader"])
    log_progress(show_progress, "batch 1/1 | loading data")
    t_data = perf_counter()
    batch_data, batch_metadata = next(iterator)
    data_elapsed = perf_counter() - t_data
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "batch_loaded",
            data_load_s=data_elapsed,
            batch_ts_shape=list(batch_data["ts"].shape),
            batch_time_delta_shape=list(batch_data["time_delta_input"].shape),
        )

    t_to_device = perf_counter()
    ts = batch_data["ts"].to(context["device"], non_blocking=True)
    time_delta_input = batch_data["time_delta_input"].to(context["device"], non_blocking=True)
    to_device_s = perf_counter() - t_to_device
    log_progress(show_progress, f"batch 1/1 | tensors input_ts={tuple(ts.shape)} time_delta_input={tuple(time_delta_input.shape)}")
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "batch_to_device",
            to_device_s=to_device_s,
            ts_dtype=str(ts.dtype),
            time_delta_dtype=str(time_delta_input.dtype),
            device=str(context["device"]),
            **_memory_stats_mb(device_type=context["device_type"], device=context["device"]),
        )

    t_io_init = perf_counter()
    io = _init_single_sample_io(
        batch_data=batch_data,
        batch_metadata=batch_metadata,
        prediction_nc_path=prediction_nc_path,
        channels=data["channels"],
        prediction_steps=data["prediction_steps"],
        prediction_dtype=data["np_dtype"],
        target_delta_minutes=data["target_delta_minutes"],
        expected_img_size=data["expected_img_size"],
        show_progress=show_progress,
        debug_logger=debug_logger,
    )
    io_init_s = perf_counter() - t_io_init
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log("io_initialized", io_init_s=io_init_s)

    step_losses, gt_pairs_used, gt_pairs_expected, infer_elapsed = _run_single_sample_rollout(
        model=context["model"],
        ts=ts,
        time_delta_input=time_delta_input,
        writer=io["writer"],
        sample_id=io["sample_id"],
        channels=data["channels"],
        prediction_steps=data["prediction_steps"],
        input_labels=io["input_labels"],
        output_labels=io["output_labels"],
        prediction_timestamps=io["prediction_timestamps"],
        inverse_batch_fn=data["inverse_batch_fn"],
        load_gt_frame_for_prediction_timestamp=data["load_gt_frame_for_prediction_timestamp"],
        prediction_dtype=data["np_dtype"],
        nan_frame_hw=io["nan_frame_hw"],
        device=context["device"],
        device_type=context["device_type"],
        amp_dtype=context["amp_dtype"],
        autocast_enabled=context["autocast_enabled"],
        gt_prefetch_workers=int(advanced_cfg["gt_prefetch_workers"]),
        show_progress=show_progress,
        debug_logger=debug_logger,
    )

    batch_loss = float(np.mean(step_losses)) if step_losses else float("nan")
    batch_loss_text = f"loss={batch_loss:.6f}" if np.isfinite(batch_loss) else "GT missing"
    log_progress(show_progress, f"batch 1/1 | done forward | {batch_loss_text} | gt_steps_used={len(step_losses)}/{data['prediction_steps']} | infer_s={infer_elapsed:.3f}")

    t_finalize = perf_counter()
    output_path = io["writer"].finalize(samples_written=1)
    finalize_s = perf_counter() - t_finalize
    log_progress(show_progress, f"saved prediction.nc | path={output_path}")
    if not np.isfinite(batch_loss):
        log_progress(show_progress, "No valid ground-truth targets available for requested rollout. GT missing; saving predictions without MSE.")
    if debug_logger is not None and debug_logger.enabled:
        debug_logger.log(
            "inference_complete",
            infer_s=infer_elapsed,
            finalize_s=finalize_s,
            avg_loss=None if not np.isfinite(batch_loss) else batch_loss,
            gt_steps_used=len(step_losses),
            prediction_steps=data["prediction_steps"],
            mode=("input_only" if gt_pairs_used == 0 else ("partial_gt" if gt_pairs_used < gt_pairs_expected else "full_gt")),
            output_path=output_path,
            **_memory_stats_mb(device_type=context["device_type"], device=context["device"]),
        )

    mode = "input_only" if gt_pairs_used == 0 else ("partial_gt" if gt_pairs_used < gt_pairs_expected else "full_gt")
    return InferenceSummary(
        avg_loss=batch_loss if np.isfinite(batch_loss) else None,
        timed_batches=1,
        avg_data_seconds=data_elapsed,
        avg_infer_seconds=infer_elapsed,
        prediction_nc_path=output_path,
        mode=mode,
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
        print(f"Inference mode       : {inference_summary.mode}")
        print("Sample selection     : first valid sample only (batch=1)")
        if inference_summary.avg_loss is None or not np.isfinite(float(inference_summary.avg_loss)):
            print("Avg rollout MSE      : GT missing")
        else:
            print(f"Avg rollout MSE      : {float(inference_summary.avg_loss):.6f}")
        print(f"Avg data sec         : {inference_summary.avg_data_seconds:.3f}")
        print(f"Avg infer sec        : {inference_summary.avg_infer_seconds:.3f}")
        print(f"Prediction file      : {inference_summary.prediction_nc_path}")
        print("GT variables         : gt_<channel> (NaN where GT is unavailable)")
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
    rollout_steps = _select_rollout_steps(
        user_cfg=user_cfg,
        cli_rollout_steps=args.rollout_steps,
        use_prompt=prompt_for_dates,
    )

    output_dir = _resolve_path(str(user_cfg["output_dir"]), config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_nc_path = output_dir / "prediction.nc"

    validation_data_dir = _resolve_path(str(advanced_cfg["validation_data_dir"]), config_dir)
    index_path = _resolve_path(str(advanced_cfg["index_path"]), config_dir)
    show_progress = bool(advanced_cfg["show_progress"])
    debug_logger = _create_debug_logger(advanced_cfg=advanced_cfg, output_dir=output_dir, config_dir=config_dir)

    print(f"Easy config          : {config_path}")
    print(
        "Download window      : "
        f"{_format_datetime(start_dt)} -> {_format_datetime(end_dt)} UTC"
    )
    print(f"Validation data dir  : {validation_data_dir}")
    print(f"Index CSV            : {index_path}")
    print(f"Prediction output    : {prediction_nc_path}")
    print(f"Rollout steps        : {int(rollout_steps)}")
    print(f"Prediction steps     : {int(rollout_steps) + 1}")
    print(
        f"Input offsets (min)  : {[int(v) for v in advanced_cfg['time_delta_input_minutes']]}"
    )
    print(f"Target delta (min)   : {int(advanced_cfg['time_delta_target_minutes'])}")
    print("Sample mode          : first valid sample only (batch=1)")
    if debug_logger.enabled:
        print(f"Debug mode           : enabled ({debug_logger.log_path})")

    try:
        if debug_logger.enabled:
            debug_logger.log(
                "run_started",
                config_path=str(config_path),
                start_datetime_utc=_format_datetime(start_dt),
                end_datetime_utc=_format_datetime(end_dt),
                rollout_steps=int(rollout_steps),
            )
        if args.dry_run:
            print("Dry run enabled. No download or inference executed.")
            return 0

        try:
            _validate_rollout_against_window(
                start_dt=start_dt,
                end_dt=end_dt,
                time_delta_input_minutes=[int(v) for v in advanced_cfg["time_delta_input_minutes"]],
                time_delta_target_minutes=int(advanced_cfg["time_delta_target_minutes"]),
                rollout_steps=int(rollout_steps),
            )
        except Exception as exc:
            print(f"ERROR rollout validation: {exc}", file=sys.stderr)
            return 1

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

        t_index = perf_counter()
        build_index_csv_for_range(
            validation_data_dir=validation_data_dir,
            index_path=index_path,
            start_datetime=start_dt,
            end_datetime=end_dt,
            cadence_minutes=int(advanced_cfg["cadence_minutes"]),
        )
        if debug_logger.enabled:
            debug_logger.log("index_built", index_build_s=perf_counter() - t_index, index_path=str(index_path))

        try:
            inference_summary = run_inference_pipeline(
                advanced_cfg=advanced_cfg,
                config_dir=config_dir,
                prediction_nc_path=prediction_nc_path,
                rollout_steps=int(rollout_steps),
                show_progress=show_progress,
                debug_logger=debug_logger,
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
    finally:
        debug_logger.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Paderborn-specific input handling; no target ground truth is read here."""

from pathlib import Path
import re

import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from thesislib.common import normalize_windows


FILENAME = re.compile(r"(N\d+_M\d+_F\d+)_(K[AIB]?\d+)_(\d+)\.mat")
HEALTHY = {f"K{i:03d}" for i in range(1, 7)}
OUTER = set("KA01 KA03 KA04 KA05 KA06 KA07 KA08 KA09 KA15 KA16 KA22 KA30".split())
INNER = set("KI01 KI03 KI04 KI05 KI07 KI08 KI14 KI16 KI17 KI18 KI21".split())
COMBINED = {"KB23", "KB24", "KB27"}
KNOWN_BEARINGS = HEALTHY | OUTER | INNER | COMBINED
ARTIFICIAL = set("KA01 KA03 KA05 KA06 KA07 KA08 KA09 KI01 KI03 KI05 KI07 KI08".split())
PAPER_URL = "https://mb.uni-paderborn.de/fileadmin-mb/kat/PDF/Veroeffentlichungen/20160703_PHME16_CM_bearing.pdf"


def validate_protocol(cfg):
    groups = cfg["bearing_groups"]
    seen = set()
    for role, bearings in groups.items():
        if len(bearings) != len(set(bearings)) or seen.intersection(bearings):
            raise ValueError(f"Duplicate/overlapping bearing IDs in {role}")
        seen.update(bearings)
        if not set(bearings).intersection(HEALTHY) or not (set(bearings) - HEALTHY):
            raise ValueError(f"Both binary classes required in {role}")
    if seen != KNOWN_BEARINGS:
        raise ValueError("Protocol must account for all 32 documented bearings.")
    if cfg["nominal_raw_fs_hz"] != 64000 or cfg["processed_fs_hz"] != 16000:
        raise ValueError("This version implements the declared 64 kHz to 16 kHz protocol only.")
    if cfg["stride"] != cfg["window_length"]:
        raise ValueError("This version requires nonoverlapping windows.")


def read_vibration(path):
    """Use channel name and its MATLAB one-based XIndex, not a fixed array slot.

    Published nominal DAQ rate is 64 kHz. Embedded HostService time axes can
    differ from an exact 64 kHz grid. We report this, never silently infer a
    replacement sample rate from them or interpolate an unverified time axis.
    """
    path = Path(path)
    data = loadmat(path, simplify_cells=True)
    if path.stem in data:
        record = data[path.stem]
    else:
        keys = [key for key in data if not key.startswith("__")]
        if len(keys) != 1:
            raise ValueError(f"Ambiguous MATLAB top-level variables: {path}")
        record = data[keys[0]]
    channels = [record["Y"]] if isinstance(record["Y"], dict) else list(record["Y"])
    candidates = [item for item in channels if item["Name"] == "vibration_1"]
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one vibration_1 channel: {path}")
    channel = candidates[0]
    x = np.asarray(channel["Data"], dtype=np.float64).reshape(-1)
    axes = [record["X"]] if isinstance(record["X"], dict) else list(record["X"])
    index = int(channel["XIndex"]) - 1
    if not 0 <= index < len(axes):
        raise ValueError(f"Invalid MATLAB XIndex: {path}")
    t = np.asarray(axes[index]["Data"], dtype=np.float64).reshape(-1)
    if x.size != t.size or x.size < 8192 or not np.isfinite(x).all() or x.std() < 1e-10:
        raise ValueError(f"Invalid/constant/short vibration recording: {path}")
    if not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
        raise ValueError(f"Invalid time axis: {path}")
    duration = float(t[-1] - t[0])
    empirical_rate = (len(x) - 1) / duration
    quality = {"samples": len(x), "rms_recorded_units": float(np.sqrt(np.mean(x ** 2))),
               "std_recorded_units": float(x.std()), "embedded_time_span": duration,
               "embedded_axis_mean_rate_if_seconds": empirical_rate,
               "embedded_axis_rate_relative_difference_from_nominal": empirical_rate / 64000 - 1,
               "nominal_fs_hz": 64000, "nominal_duration_s": len(x) / 64000,
               "sampling_rate_policy": "published nominal 64 kHz; embedded timing diagnostic only"}
    return x, quality


def make_windows(raw, cfg):
    processed = resample_poly(raw, 1, 4)
    trim, length = cfg["processed_edge_trim_samples"], cfg["window_length"]
    if trim:
        processed = processed[trim:-trim]
    count = len(processed) // length
    if count < 1:
        raise ValueError("Recording too short for a complete window.")
    windows = processed[:count * length].reshape(count, length)
    if not np.isfinite(windows).all() or np.any(windows.std(axis=1) < 1e-10):
        raise ValueError("Invalid/constant signal window.")
    return normalize_windows(windows)[:, None, :]


def bearing_metadata(bearing):
    """Curator/evaluator metadata, not called by training or prediction."""
    if bearing not in KNOWN_BEARINGS:
        raise ValueError(f"Unknown bearing: {bearing}")
    if bearing in HEALTHY:
        return {"class_id": 0, "damage_family": "healthy", "damage_origin": "healthy"}
    family = "outer_ring" if bearing in OUTER else "inner_ring" if bearing in INNER else "combined_inner_outer"
    return {"class_id": 1, "damage_family": family,
            "damage_origin": "artificial" if bearing in ARTIFICIAL else "accelerated_lifetime"}
